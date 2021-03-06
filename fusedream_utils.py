import torch
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision
import BigGAN_utils.utils as utils
import clip
import torch.nn.functional as F
from DiffAugment_pytorch import DiffAugment
import numpy as np
import lpips

LATENT_NOISE = 0.01
Z_THRES = 2.0
POLICY = 'color,translation,resize,cutout'
TEST_POLICY = 'color,translation,resize,cutout'
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

def AugmentLoss(img, clip_model, text, replicate=10, interp_mode='bilinear', policy=POLICY):

    clip_c = clip_model.logit_scale.exp()
    img_aug = DiffAugment(img.repeat(replicate, 1, 1, 1), policy=policy)
    img_aug = (img_aug+1.)/2.
    img_aug = F.interpolate(img_aug, size=224, mode=interp_mode)
    img_aug.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

    logits_per_image, logits_per_text = clip_model(img_aug, text)
    logits_per_image = logits_per_image / clip_c
    concept_loss = (-1.) * logits_per_image 
     
    return concept_loss.mean(dim=0, keepdim=False)

def NaiveSemanticLoss(img, clip_model, text, interp_mode='bilinear'):

    clip_c = clip_model.logit_scale.exp()
    img = (img+1.)/2.
    img = F.interpolate(img, size=224, mode=interp_mode)
    img.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

    logits_per_image, logits_per_text = clip_model(img, text)
    logits_per_image = logits_per_image / clip_c
    concept_loss = (-1.) * logits_per_image 
     
    return concept_loss.mean(dim=0, keepdim=False)

def get_gaussian_mask(size=256):
    x, y = np.meshgrid(np.linspace(-1,1, size), np.linspace(-1,1,size))
    dst = np.sqrt(x*x+y*y)
      
    # Intializing sigma and muu
    sigma = 1
    muu = 0.000
      
    # Calculating Gaussian array
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    
    return gauss

def save_image(img, path, n_per_row=1):
    with torch.no_grad():
        torchvision.utils.save_image(
            torch.from_numpy(img.cpu().numpy()), ##hack, to turn Distribution back to tensor
            path,
            nrow=n_per_row,
            normalize=True,
        )

def get_G(resolution=256):
    if resolution == 256:
        parser = utils.prepare_parser()
        parser = utils.add_sample_parser(parser)
        config = vars(parser.parse_args())

        # See: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/scripts/sample_BigGAN_bs256x8.sh.
        config["resolution"] = utils.imsize_dict["I128_hdf5"]
        config["n_classes"] = utils.nclass_dict["I128_hdf5"]
        config["G_activation"] = utils.activation_dict["inplace_relu"]
        config["D_activation"] = utils.activation_dict["inplace_relu"]
        config["G_attn"] = "128"
        config["D_attn"] = "128"
        config["G_ch"] = 96
        config["D_ch"] = 96
        config["hier"] = True
        config["dim_z"] = 140
        config["shared_dim"] = 128
        config["G_shared"] = True
        config = utils.update_config_roots(config)
        config["skip_init"] = True
        config["no_optim"] = True
        config["device"] = "cuda"
        config["resolution"] = 256

        # Set up cudnn.benchmark for free speed.
        torch.backends.cudnn.benchmark = True

        # Import the model.
        model = __import__(config["model"])
        G = model.Generator(**config).to(config["device"])
        utils.count_parameters(G)

        # Load weights.
        weights_path = "./BigGAN_utils/weights/biggan-256.pth"  # Change this.
        G.load_state_dict(torch.load(weights_path), strict=False)
    elif resolution == 512:
        parser = utils.prepare_parser()
        parser = utils.add_sample_parser(parser)
        config = vars(parser.parse_args())

        # See: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/scripts/sample_BigGAN_bs128x8.sh.
        config["resolution"] = 512
        config["n_classes"] = utils.nclass_dict["I128_hdf5"]
        config["G_activation"] = utils.activation_dict["inplace_relu"]
        config["D_activation"] = utils.activation_dict["inplace_relu"]
        config["G_attn"] = "64"
        config["D_attn"] = "64"
        config["G_ch"] = 96
        config["D_ch"] = 64
        config["hier"] = True
        config["dim_z"] = 128
        config["shared_dim"] = 128
        config["G_shared"] = True
        config = utils.update_config_roots(config)
        config["skip_init"] = True
        config["no_optim"] = True
        config["device"] = "cuda"

        # Set up cudnn.benchmark for free speed.
        torch.backends.cudnn.benchmark = True

        # Import the model.
        model = __import__(config["model"])
        #print(config["model"])
        G = model.Generator(**config).to(config["device"])
        utils.count_parameters(G)
        #print('G parameters:')
        #for p, m in G.named_parameters():
        #    print(p)
        # Load weights.
        weights_path = "./BigGAN_utils/weights/biggan-512.pth"  # Change this.
        G.load_state_dict(torch.load(weights_path), strict=False)

    return G, config

class FuseDreamBaseGenerator():
    def __init__(self, G, G_config, G_batch_size=10, clip_mode="ViT-B/32", interp_mode='bilinear'):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.G = G
        self.clip_model, _ = clip.load(clip_mode, device=device) 
        
        (self.z_, self.y_) = utils.prepare_z_y(
            G_batch_size,
            self.G.dim_z,
            G_config["n_classes"],
            device=G_config["device"],
            fp16=G_config["G_fp16"],
            z_var=G_config["z_var"],
        )

        self.G.eval()

        for p in self.G.parameters():
            p.requires_grad = False
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.interp_mode = interp_mode 
  
    def generate_basis(self, text, init_iters=500, num_basis=5):
        
        #restriction to have basis> no. of nouns
        
        #new commit
        imageNet = [ "tench,Tinca,tinca" ,
            "goldfish, Carassius auratus" ,
            "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias" ,
            "tiger shark, Galeocerdo cuvieri" ,
            "hammerhead, hammerhead shark" ,
            "electric ray, crampfish, numbfish, torpedo" ,
            "stingray" ,
            "cock" ,
            "hen" ,
            "ostrich, Struthio camelus",
            "brambling, Fringilla montifringilla" ,
            "goldfinch, Carduelis carduelis" ,
            "house finch, linnet, Carpodacus mexicanus" ,
            "junco, snowbird" ,
            "indigo bunting, indigo finch, indigo bird, Passerina cyanea" ,
            "robin, American robin, Turdus migratorius" ,
            "bulbul" ,
            "jay" ,
            "magpie" ,
            "chickadee" ,
            "water ouzel, dipper" ,
            "kite" ,
            "bald eagle, American eagle, Haliaeetus leucocephalus" ,
            "vulture" ,
            "great grey owl, great gray owl, Strix nebulosa" ,
            "European fire salamander, Salamandra salamandra" ,
            "common newt, Triturus vulgaris" ,
            "eft" ,
            "spotted salamander, Ambystoma maculatum" ,
            "axolotl, mud puppy, Ambystoma mexicanum" ,
            "bullfrog, Rana catesbeiana" ,
            "tree frog, tree-frog" ,
            "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui" ,
            "loggerhead, loggerhead turtle, Caretta caretta" ,
            "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea" ,
            "mud turtle" ,
            "terrapin" ,
            "box turtle, box tortoise" ,
            "banded gecko" ,
            "common iguana, iguana, Iguana iguana" ,
            "American chameleon, anole, Anolis carolinensis" ,
            "whiptail, whiptail lizard",
            "agama" ,
            "frilled lizard, Chlamydosaurus kingi" ,
            "alligator lizard",
            "Gila monster, Heloderma suspectum" ,
            "green lizard, Lacerta viridis" ,
            "African chameleon, Chamaeleo chamaeleon" ,
            "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis" ,
            "African crocodile, Nile crocodile, Crocodylus niloticus" ,
            "American alligator, Alligator mississipiensis",
            "triceratops",
            "thunder snake, worm snake, Carphophis amoenus",
            "ringneck snake, ring-necked snake, ring snake",
            "hognose snake, puff adder, sand viper",
            "green snake, grass snake",
            "king snake, kingsnake",
            "garter snake, grass snake",
            "water snake",
            "vine snake",
            "night snake, Hypsiglena torquata",
            "boa constrictor, Constrictor constrictor",
            "rock python, rock snake, Python sebae",
            "Indian cobra, Naja naja",
            "green mamba",
            "sea snake",
            "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus",
            "diamondback, diamondback rattlesnake, Crotalus adamanteus",
            "sidewinder, horned rattlesnake, Crotalus cerastes",
            "trilobite",
            "harvestman, daddy longlegs, Phalangium opilio",
            "scorpion",
            "black and gold garden spider, Argiope aurantia",
            "barn spider, Araneus cavaticus",
            "garden spider, Aranea diademata",
            "black widow, Latrodectus mactans",
            "tarantula",
            "wolf spider, hunting spider",
            "tick",
            "centipede",
            "black grouse",
            "ptarmigan",
            "ruffed grouse, partridge, Bonasa umbellus",
            "prairie chicken, prairie grouse, prairie fowl",
            "peacock",
            "quail",
            "partridge",
            "African grey, African gray, Psittacus erithacus",
            "macaw",
            "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
            "lorikeet",
            "coucal",
            "bee eater",
            "hornbill",
            "hummingbird",
            "jacamar",
            "toucan",
            "drake",
            "red-breasted merganser, Mergus serrator",
            "goose",
            "black swan, Cygnus atratus",
            "tusker",
            "echidna, spiny anteater, anteater",
            "platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus",
            "wallaby, brush kangaroo",
            "koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus",
            "wombat",
            "jellyfish",
            "sea anemone, anemone",
            "brain coral",
            "flatworm, platyhelminth",
            "nematode, nematode worm, roundworm",
            "conch",
            "snail",
            "slug",
            "sea slug, nudibranch",
            "chiton, coat-of-mail shell, sea cradle, polyplacophore",
            "chambered nautilus, pearly nautilus, nautilus",
            "Dungeness crab, Cancer magister",
            "rock crab, Cancer irroratus",
            "fiddler crab",
            "king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica",
            "American lobster, Northern lobster, Maine lobster, Homarus americanus",
            "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
            "crayfish, crawfish, crawdad, crawdaddy",
            "hermit crab",
            "isopod",
            "white stork, Ciconia ciconia",
            "black stork, Ciconia nigra",
            "spoonbill",
            "flamingo",
            "little blue heron, Egretta caerulea",
            "American egret, great white heron, Egretta albus",
            "bittern",
            "crane",
            "limpkin, Aramus pictus",
            "European gallinule, Porphyrio porphyrio",
            "American coot, marsh hen, mud hen, water hen, Fulica americana",
            "bustard",
            "ruddy turnstone, Arenaria interpres",
            "red-backed sandpiper, dunlin, Erolia alpina",
            "redshank, Tringa totanus",
            "dowitcher",
            "oystercatcher, oyster catcher",
            "pelican",
            "king penguin, Aptenodytes patagonica",
            "albatross, mollymawk",
            "grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus",
            "killer whale, killer, orca, grampus, sea wolf, Orcinus orca",
            "dugong, Dugong dugon",
            "sea lion",
            "Chihuahua",
            "Japanese spaniel",
            "Maltese dog, Maltese terrier, Maltese",
            "Pekinese, Pekingese, Peke",
            "Shih-Tzu",
            "Blenheim spaniel",
            "papillon",
            "toy terrier",
            "Rhodesian ridgeback",
            "Afghan hound, Afghan",
            "basset, basset hound",
            "beagle",
            "bloodhound, sleuthhound",
            "bluetick",
            "black-and-tan coonhound",
            "Walker hound, Walker foxhound",
            "English foxhound",
            "redbone",
            "borzoi, Russian wolfhound",
            "Irish wolfhound",
            "Italian greyhound",
            "whippet",
            "Ibizan hound, Ibizan Podenco",
            "Norwegian elkhound, elkhound",
            "otterhound, otter hound",
            "Saluki, gazelle hound",
            "Scottish deerhound, deerhound",
            "Weimaraner",
            "Staffordshire bullterrier, Staffordshire bull terrier",
            "American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier",
            "Bedlington terrier",
            "Border terrier",
            "Kerry blue terrier",
            "Irish terrier",
            "Norfolk terrier",
            "Norwich terrier",
            "Yorkshire terrier",
            "wire-haired fox terrier",
            "Lakeland terrier",
            "Sealyham terrier, Sealyham",
            "Airedale, Airedale terrier",
            "cairn, cairn terrier",
            "Australian terrier",
            "Dandie Dinmont, Dandie Dinmont terrier",
            "Boston bull, Boston terrier",
            "miniature schnauzer",
            "giant schnauzer",
            "standard schnauzer",
            "Scotch terrier, Scottish terrier, Scottie",
            "Tibetan terrier, chrysanthemum dog",
            "silky terrier, Sydney silky",
            "soft-coated wheaten terrier",
            "West Highland white terrier",
            "Lhasa, Lhasa apso",
            "flat-coated retriever",
            "curly-coated retriever",
            "golden retriever",
            "Labrador retriever",
            "Chesapeake Bay retriever",
            "German short-haired pointer",
            "vizsla, Hungarian pointer",
            "English setter",
            "Irish setter, red setter",
            "Gordon setter",
            "Brittany spaniel",
            "clumber, clumber spaniel",
            "English springer, English springer spaniel",
            "Welsh springer spaniel",
            "cocker spaniel, English cocker spaniel, cocker",
            "Sussex spaniel",
            "Irish water spaniel",
            "kuvasz",
            "schipperke",
            "groenendael",
            "malinois",
            "briard",
            "kelpie",
            "komondor",
            "Old English sheepdog, bobtail",
            "Shetland sheepdog, Shetland sheep dog, Shetland",
            "collie",
            "Border collie",
            "Bouvier des Flandres, Bouviers des Flandres",
            "Rottweiler",
            "German shepherd, German shepherd dog, German police dog, alsatian",
            "Doberman, Doberman pinscher",
            "miniature pinscher",
            "Greater Swiss Mountain dog",
            "Bernese mountain dog",
            "Appenzeller",
            "EntleBucher",
            "boxer",
            "bull mastiff",
            "Tibetan mastiff",
            "French bulldog",
            "Great Dane",
            "Saint Bernard, St Bernard",
            "Eskimo dog, husky",
            "malamute, malemute, Alaskan malamute",
            "Siberian husky",
            "dalmatian, coach dog, carriage dog",
            "affenpinscher, monkey pinscher, monkey dog",
            "basenji",
            "pug, pug-dog",
            "Leonberg",
            "Newfoundland, Newfoundland dog",
            "Great Pyrenees",
            "Samoyed, Samoyede",
            "Pomeranian",
            "chow, chow chow",
            "keeshond",
            "Brabancon griffon",
            "Pembroke, Pembroke Welsh corgi",
            "Cardigan, Cardigan Welsh corgi",
            "toy poodle",
            "miniature poodle",
            "standard poodle",
            "Mexican hairless",
            "timber wolf, grey wolf, gray wolf, Canis lupus",
            "white wolf, Arctic wolf, Canis lupus tundrarum",
            "red wolf, maned wolf, Canis rufus, Canis niger",
            "coyote, prairie wolf, brush wolf, Canis latrans",
            "dingo, warrigal, warragal, Canis dingo",
            "dhole, Cuon alpinus",
            "African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus",
            "hyena, hyaena",
            "red fox, Vulpes vulpes",
            "kit fox, Vulpes macrotis",
            "Arctic fox, white fox, Alopex lagopus",
            "grey fox, gray fox, Urocyon cinereoargenteus",
            "tabby, tabby cat",
            "tiger cat",
            "Persian cat",
            "Siamese cat, Siamese",
            "Egyptian cat",
            "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
            "lynx, catamount",
            "leopard, Panthera pardus",
            "snow leopard, ounce, Panthera uncia",
            "jaguar, panther, Panthera onca, Felis onca",
            "lion, king of beasts, Panthera leo",
            "tiger, Panthera tigris",
            "cheetah, chetah, Acinonyx jubatus",
            "brown bear, bruin, Ursus arctos",
            "American black bear, black bear, Ursus americanus, Euarctos americanus",
            "ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus",
            "sloth bear, Melursus ursinus, Ursus ursinus",
            "mongoose",
            "meerkat, mierkat",
            "tiger beetle",
            "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
            "ground beetle, carabid beetle",
            "long-horned beetle, longicorn, longicorn beetle",
            "leaf beetle, chrysomelid",
            "dung beetle",
            "rhinoceros beetle",
            "weevil",
            "fly",
            "bee",
            "ant, emmet, pismire",
            "grasshopper, hopper",
            "cricket",
            "walking stick, walkingstick, stick insect",
            "cockroach, roach",
            "mantis, mantid",
            "cicada, cicala",
            "leafhopper",
            "lacewing, lacewing fly",
            "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
            "damselfly",
            "admiral",
            "ringlet, ringlet butterfly",
            "monarch, monarch butterfly, milkweed butterfly, Danaus plexippus",
            "cabbage butterfly",
            "sulphur butterfly, sulfur butterfly",
            "lycaenid, lycaenid butterfly",
            "starfish, sea star",
            "sea urchin",
            "sea cucumber, holothurian",
            "wood rabbit, cottontail, cottontail rabbit",
            "hare",
            "Angora, Angora rabbit",
            "hamster",
            "porcupine, hedgehog",
            "fox squirrel, eastern fox squirrel, Sciurus niger",
            "marmot",
            "beaver",
            "guinea pig, Cavia cobaya",
            "sorrel",
            "zebra",
            "hog, pig, grunter, squealer, Sus scrofa",
            "wild boar, boar, Sus scrofa",
            "warthog",
            "hippopotamus, hippo, river horse, Hippopotamus amphibius",
            "ox",
            "water buffalo, water ox, Asiatic buffalo, Bubalus bubalis",
            "bison",
            "ram, tup",
            "bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis",
            "ibex, Capra ibex",
            "hartebeest",
            "impala, Aepyceros melampus",
            "gazelle",
            "Arabian camel, dromedary, Camelus dromedarius",
            "llama",
            "weasel",
            "mink",
            "polecat, fitch, foulmart, foumart, Mustela putorius",
            "black-footed ferret, ferret, Mustela nigripes",
            "otter",
            "skunk, polecat, wood pussy",
            "badger",
            "armadillo",
            "three-toed sloth, ai, Bradypus tridactylus",
            "orangutan, orang, orangutang, Pongo pygmaeus",
            "gorilla, Gorilla gorilla",
            "chimpanzee, chimp, Pan troglodytes",
            "gibbon, Hylobates lar",
            "siamang, Hylobates syndactylus, Symphalangus syndactylus",
            "guenon, guenon monkey",
            "patas, hussar monkey, Erythrocebus patas",
            "baboon",
            "macaque",
            "langur",
            "colobus, colobus monkey",
            "proboscis monkey, Nasalis larvatus",
            "marmoset",
            "capuchin, ringtail, Cebus capucinus",
            "howler monkey, howler",
            "titi, titi monkey",
            "spider monkey, Ateles geoffroyi",
            "squirrel monkey, Saimiri sciureus",
            "Madagascar cat, ring-tailed lemur, Lemur catta",
            "indri, indris, Indri indri, Indri brevicaudatus",
            "Indian elephant, Elephas maximus",
            "African elephant, Loxodonta africana",
            "lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens",
            "giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca",
            "barracouta, snoek",
            "eel",
            "coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch",
            "rock beauty, Holocanthus tricolor",
            "anemone fish",
            "sturgeon",
            "gar, garfish, garpike, billfish, Lepisosteus osseus",
            "lionfish",
            "puffer, pufferfish, blowfish, globefish",
            "abacus",
            "abaya",
            "academic gown, academic robe, judge's robe",
            "accordion, piano accordion, squeeze box",
            "acoustic guitar",
            "aircraft carrier, carrier, flattop, attack aircraft carrier",
            "airliner",
            "airship, dirigible",
            "altar",
            "ambulance",
            "amphibian, amphibious vehicle",
            "analog clock",
            "apiary, bee house",
            "apron",
            "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
            "assault rifle, assault gun",
            "backpack, back pack, knapsack, packsack, rucksack, haversack",
            "bakery, bakeshop, bakehouse",
            "balance beam, beam",
            "balloon",
            "ballpoint, ballpoint pen, ballpen, Biro",
            "Band Aid",
            "banjo",
            "bannister, banister, balustrade, balusters, handrail",
            "barbell",
            "barber chair",
            "barbershop",
            "barn",
            "barometer",
            "barrel, cask",
            "barrow, garden cart, lawn cart, wheelbarrow",
            "baseball",
            "basketball",
            "bassinet",
            "bassoon",
            "bathing cap, swimming cap",
            "bath towel",
            "bathtub, bathing tub, bath, tub",
            "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon",
            "beacon, lighthouse, beacon light, pharos",
            "beaker",
            "bearskin, busby, shako",
            "beer bottle",
            "beer glass",
            "bell cote, bell cot",
            "bib",
            "bicycle-built-for-two, tandem bicycle, tandem",
            "bikini, two-piece",
            "binder, ring-binder",
            "binoculars, field glasses, opera glasses",
            "birdhouse",
            "boathouse",
            "bobsled, bobsleigh, bob",
            "bolo tie, bolo, bola tie, bola",
            "bonnet, poke bonnet",
            "bookcase",
            "bookshop, bookstore, bookstall",
            "bottlecap",
            "bow",
            "bow tie, bow-tie, bowtie",
            "brass, memorial tablet, plaque",
            "brassiere, bra, bandeau",
            "breakwater, groin, groyne, mole, bulwark, seawall, jetty",
            "breastplate, aegis, egis",
            "broom",
            "bucket, pail",
            "buckle",
            "bulletproof vest",
            "bullet train, bullet",
            "butcher shop, meat market",
            "cab, hack, taxi, taxicab",
            "caldron, cauldron",
            "candle, taper, wax light",
            "cannon",
            "canoe",
            "can opener, tin opener",
            "cardigan",
            "car mirror",
            "carousel, carrousel, merry-go-round, roundabout, whirligig",
            "carpenter's kit, tool kit",
            "carton",
            "car wheel",
            "cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM",
            "cassette",
            "cassette player",
            "castle",
            "catamaran",
            "CD player",
            "cello, violoncello",
            "cellular telephone, cellular phone, cellphone, cell, mobile phone",
            "chain",
            "chainlink fence",
            "chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour",
            "chain saw, chainsaw",
            "chest",
            "chiffonier, commode",
            "chime, bell, gong",
            "china cabinet, china closet",
            "Christmas stocking",
            "church, church building",
            "cinema, movie theater, movie theatre, movie house, picture palace",
            "cleaver, meat cleaver, chopper",
            "cliff dwelling",
            "cloak",
            "clog, geta, patten, sabot",
            "cocktail shaker",
            "coffee mug",
            "coffeepot",
            "coil, spiral, volute, whorl, helix",
            "combination lock",
            "computer keyboard, keypad",
            "confectionery, confectionary, candy store",
            "container ship, containership, container vessel",
            "convertible",
            "corkscrew, bottle screw",
            "cornet, horn, trumpet, trump",
            "cowboy boot",
            "cowboy hat, ten-gallon hat",
            "cradle",
            "crane",
            "crash helmet",
            "crate",
            "crib, cot",
            "Crock Pot",
            "croquet ball",
            "crutch",
            "cuirass",
            "dam, dike, dyke",
            "desk",
            "desktop computer",
            "dial telephone, dial phone",
            "diaper, nappy, napkin",
            "digital clock",
            "digital watch",
            "dining table, board",
            "dishrag, dishcloth",
            "dishwasher, dish washer, dishwashing machine",
            "disk brake, disc brake",
            "dock, dockage, docking facility",
            "dogsled, dog sled, dog sleigh",
            "dome",
            "doormat, welcome mat",
            "drilling platform, offshore rig",
            "drum, membranophone, tympan",
            "drumstick",
            "dumbbell",
            "Dutch oven",
            "electric fan, blower",
            "electric guitar",
            "electric locomotive",
            "entertainment center",
            "envelope",
            "espresso maker",
            "face powder",
            "feather boa, boa",
            "file, file cabinet, filing cabinet",
            "fireboat",
            "fire engine, fire truck",
            "fire screen, fireguard",
            "flagpole, flagstaff",
            "flute, transverse flute",
            "folding chair",
            "football helmet",
            "forklift",
            "fountain",
            "fountain pen",
            "four-poster",
            "freight car",
            "French horn, horn",
            "frying pan, frypan, skillet",
            "fur coat",
            "garbage truck, dustcart",
            "gasmask, respirator, gas helmet",
            "gas pump, gasoline pump, petrol pump, island dispenser",
            "goblet",
            "go-kart",
            "golf ball",
            "golfcart, golf cart",
            "gondola",
            "gong, tam-tam",
            "gown",
            "grand piano, grand",
            "greenhouse, nursery, glasshouse",
            "grille, radiator grille",
            "grocery store, grocery, food market, market",
            "guillotine",
            "hair slide",
            "hair spray",
            "half track",
            "hammer",
            "hamper",
            "hand blower, blow dryer, blow drier, hair dryer, hair drier",
            "hand-held computer, hand-held microcomputer",
            "handkerchief, hankie, hanky, hankey",
            "hard disc, hard disk, fixed disk",
            "harmonica, mouth organ, harp, mouth harp",
            "harp",
            "harvester, reaper",
            "hatchet",
            "holster",
            "home theater, home theatre",
            "honeycomb",
            "hook, claw",
            "hoopskirt, crinoline",
            "horizontal bar, high bar",
            "horse cart, horse-cart",
            "hourglass",
            "iPod",
            "iron, smoothing iron",
            "jack-o'-lantern",
            "jean, blue jean, denim",
            "jeep, landrover",
            "jersey, T-shirt, tee shirt",
            "jigsaw puzzle",
            "jinrikisha, ricksha, rickshaw",
            "joystick",
            "kimono",
            "knee pad",
            "knot",
            "lab coat, laboratory coat",
            "ladle",
            "lampshade, lamp shade",
            "laptop, laptop computer",
            "lawn mower, mower",
            "lens cap, lens cover",
            "letter opener, paper knife, paperknife",
            "library",
            "lifeboat",
            "lighter, light, igniter, ignitor",
            "limousine, limo",
            "liner, ocean liner",
            "lipstick, lip rouge",
            "Loafer",
            "lotion",
            "loudspeaker, speaker, speaker unit, loudspeaker system, speaker system",
            "loupe, jeweler's loupe",
            "lumbermill, sawmill",
            "magnetic compass",
            "mailbag, postbag",
            "mailbox, letter box",
            "maillot",
            "maillot, tank suit",
            "manhole cover",
            "maraca",
            "marimba, xylophone",
            "mask",
            "matchstick",
            "maypole",
            "maze, labyrinth",
            "measuring cup",
            "medicine chest, medicine cabinet",
            "megalith, megalithic structure",
            "microphone, mike",
            "microwave, microwave oven",
            "military uniform",
            "milk can",
            "minibus",
            "miniskirt, mini",
            "minivan",
            "missile",
            "mitten",
            "mixing bowl",
            "mobile home, manufactured home",
            "Model T",
            "modem",
            "monastery",
            "monitor",
            "moped",
            "mortar",
            "mortarboard",
            "mosque",
            "mosquito net",
            "motor scooter, scooter",
            "mountain bike, all-terrain bike, off-roader",
            "mountain tent",
            "mouse, computer mouse",
            "mousetrap",
            "moving van",
            "muzzle",
            "nail",
            "neck brace",
            "necklace",
            "nipple",
            "notebook, notebook computer",
            "obelisk",
            "oboe, hautboy, hautbois",
            "ocarina, sweet potato",
            "odometer, hodometer, mileometer, milometer",
            "oil filter",
            "organ, pipe organ",
            "oscilloscope, scope, cathode-ray oscilloscope, CRO",
            "overskirt",
            "oxcart",
            "oxygen mask",
            "packet",
            "paddle, boat paddle",
            "paddlewheel, paddle wheel",
            "padlock",
            "paintbrush",
            "pajama, pyjama, pj's, jammies",
            "palace",
            "panpipe, pandean pipe, syrinx",
            "paper towel",
            "parachute, chute",
            "parallel bars, bars",
            "park bench",
            "parking meter",
            "passenger car, coach, carriage",
            "patio, terrace",
            "pay-phone, pay-station",
            "pedestal, plinth, footstall",
            "pencil box, pencil case",
            "pencil sharpener",
            "perfume, essence",
            "Petri dish",
            "photocopier",
            "pick, plectrum, plectron",
            "pickelhaube",
            "picket fence, paling",
            "pickup, pickup truck",
            "pier",
            "piggy bank, penny bank",
            "pill bottle",
            "pillow",
            "ping-pong ball",
            "pinwheel",
            "pirate, pirate ship",
            "pitcher, ewer",
            "plane, carpenter's plane, woodworking plane",
            "planetarium",
            "plastic bag",
            "plate rack",
            "plow, plough",
            "plunger, plumber's helper",
            "Polaroid camera, Polaroid Land camera",
            "pole",
            "police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria",
            "poncho",
            "pool table, billiard table, snooker table",
            "pop bottle, soda bottle",
            "pot, flowerpot",
            "potter's wheel",
            "power drill",
            "prayer rug, prayer mat",
            "printer",
            "prison, prison house",
            "projectile, missile",
            "projector",
            "puck, hockey puck",
            "punching bag, punch bag, punching ball, punchball",
            "purse",
            "quill, quill pen",
            "quilt, comforter, comfort, puff",
            "racer, race car, racing car",
            "racket, racquet",
            "radiator",
            "radio, wireless",
            "radio telescope, radio reflector",
            "rain barrel",
            "recreational vehicle, RV, R.V.",
            "reel",
            "reflex camera",
            "refrigerator, icebox",
            "remote control, remote",
            "restaurant, eating house, eating place, eatery",
            "revolver, six-gun, six-shooter",
            "rifle",
            "rocking chair, rocker",
            "rotisserie",
            "rubber eraser, rubber, pencil eraser",
            "rugby ball",
            "rule, ruler",
            "running shoe",
            "safe",
            "safety pin",
            "saltshaker, salt shaker",
            "sandal",
            "sarong",
            "sax, saxophone",
            "scabbard",
            "scale, weighing machine",
            "school bus",
            "schooner",
            "scoreboard",
            "screen, CRT screen",
            "screw",
            "screwdriver",
            "seat belt, seatbelt",
            "sewing machine",
            "shield, buckler",
            "shoe shop, shoe-shop, shoe store",
            "shoji",
            "shopping basket",
            "shopping cart",
            "shovel",
            "shower cap",
            "shower curtain",
            "ski",
            "ski mask",
            "sleeping bag",
            "slide rule, slipstick",
            "sliding door",
            "slot, one-armed bandit",
            "snorkel",
            "snowmobile",
            "snowplow, snowplough",
            "soap dispenser",
            "soccer ball",
            "sock",
            "solar dish, solar collector, solar furnace",
            "sombrero",
            "soup bowl",
            "space bar",
            "space heater",
            "space shuttle",
            "spatula",
            "speedboat",
            "spider web, spider's web",
            "spindle",
            "sports car, sport car",
            "spotlight, spot",
            "stage",
            "steam locomotive",
            "steel arch bridge",
            "steel drum",
            "stethoscope",
            "stole",
            "stone wall",
            "stopwatch, stop watch",
            "stove",
            "strainer",
            "streetcar, tram, tramcar, trolley, trolley car",
            "stretcher",
            "studio couch, day bed",
            "stupa, tope",
            "submarine, pigboat, sub, U-boat",
            "suit, suit of clothes",
            "sundial",
            "sunglass",
            "sunglasses, dark glasses, shades",
            "sunscreen, sunblock, sun blocker",
            "suspension bridge",
            "swab, swob, mop",
            "sweatshirt",
            "swimming trunks, bathing trunks",
            "swing",
            "switch, electric switch, electrical switch",
            "syringe",
            "table lamp",
            "tank, army tank, armored combat vehicle, armoured combat vehicle",
            "tape player",
            "teapot",
            "teddy, teddy bear",
            "television, television system",
            "tennis ball",
            "thatch, thatched roof",
            "theater curtain, theatre curtain",
            "thimble",
            "thresher, thrasher, threshing machine",
            "throne",
            "tile roof",
            "toaster",
            "tobacco shop, tobacconist shop, tobacconist",
            "toilet seat",
            "torch",
            "totem pole",
            "tow truck, tow car, wrecker",
            "toyshop",
            "tractor",
            "trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi",
            "tray",
            "trench coat",
            "tricycle, trike, velocipede",
            "trimaran",
            "tripod",
            "triumphal arch",
            "trolleybus, trolley coach, trackless trolley",
            "trombone",
            "tub, vat",
            "turnstile",
            "typewriter keyboard",
            "umbrella",
            "unicycle, monocycle",
            "upright, upright piano",
            "vacuum, vacuum cleaner",
            "vase",
            "vault",
            "velvet",
            "vending machine",
            "vestment",
            "viaduct",
            "violin, fiddle",
            "volleyball",
            "waffle iron",
            "wall clock",
            "wallet, billfold, notecase, pocketbook",
            "wardrobe, closet, press",
            "warplane, military plane",
            "washbasin, handbasin, washbowl, lavabo, wash-hand basin",
            "washer, automatic washer, washing machine",
            "water bottle",
            "water jug",
            "water tower",
            "whiskey jug",
            "whistle",
            "wig",
            "window screen",
            "window shade",
            "Windsor tie",
            "wine bottle",
            "wing",
            "wok",
            "wooden spoon",
            "wool, woolen, woollen",
            "worm fence, snake fence, snake-rail fence, Virginia fence",
            "wreck",
            "yawl",
            "yurt",
            "web site, website, internet site, site",
            "comic book",
            "crossword puzzle, crossword",
            "street sign",
            "traffic light, traffic signal, stoplight",
            "book jacket, dust cover, dust jacket, dust wrapper",
            "menu",
            "plate",
            "guacamole",
            "consomme",
            "hot pot, hotpot",
            "trifle",
            "ice cream, icecream",
            "ice lolly, lolly, lollipop, popsicle",
            "French loaf",
            "bagel, beigel",
            "pretzel",
            "cheeseburger",
            "hotdog, hot dog, red hot",
            "mashed potato",
            "head cabbage",
            "broccoli",
            "cauliflower",
            "zucchini, courgette",
            "spaghetti squash",
            "acorn squash",
            "butternut squash",
            "cucumber, cuke",
            "artichoke, globe artichoke",
            "bell pepper",
            "cardoon",
            "mushroom",
            "Granny Smith",
            "strawberry",
            "orange",
            "lemon",
            "fig",
            "pineapple, ananas",
            "banana",
            "jackfruit, jak, jack",
            "custard apple",
            "pomegranate",
            "hay",
            "carbonara",
            "chocolate sauce, chocolate syrup",
            "dough",
            "meat loaf, meatloaf",
            "pizza, pizza pie",
            "potpie",
            "burrito",
            "red wine",
            "espresso",
            "cup",
            "eggnog",
            "alp",
            "bubble",
            "cliff, drop, drop-off",
            "coral reef",
            "geyser",
            "lakeside, lakeshore",
            "promontory, headland, head, foreland",
            "sandbar, sand bar",
            "seashore, coast, seacoast, sea-coast",
            "valley, vale",
            "volcano",
            "ballplayer, baseball player",
            "groom, bridegroom",
            "scuba diver",
            "rapeseed",
            "daisy",
            "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
            "corn",
            "acorn",
            "hip, rose hip, rosehip",
            "buckeye, horse chestnut, conker",
            "coral fungus",
            "agaric",
            "gyromitra",
            "stinkhorn, carrion fungus",
            "earthstar",
            "hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa",
            "bolete",
            "ear, spike, capitulum",
            "toilet tissue, toilet paper, bathroom tissue"]
           
        import re
        def f(s, pat):
            pat = r'(\w*%s\w*)' % pat       # Not thrilled about this line
            return (re.findall(pat, s))

        import nltk
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        
        lines = text
        # function to test if something is a noun
        is_noun = lambda pos: pos[:2] == 'NN'
        # do the nlp stuff
        tokenized = nltk.word_tokenize(lines)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
        print (nouns)
             
        nounsArrImageNet=[]
        endIndexNoun=[]

        for noun in nouns:
            for i,val in enumerate(imageNet):
                if (f(val,noun)):
                    nounsArrImageNet.append(i)
            endIndexNoun.append(len(nounsArrImageNet)-1)
        print(nounsArrImageNet)
        print(endIndexNoun)

        
        
        print("BASIS_first_line")
        text_tok = clip.tokenize([text]).to(self.device)
        print("finish texttok and it is:")
        print(text_tok)
        clip_c = self.clip_model.logit_scale.exp() 
        print("finish clip_c and it is:")
        print(clip_c)
       
        
        
        z_init_cllt = []
        y_init_cllt = []
        z_init = None
        y_init = None
        score_init = None
        with torch.no_grad():
            for i in tqdm(range(init_iters)):
                self.z_.sample_()
                sampleembedding1,sampleembedding2=self.y_.sample_()
                print("sample_emb1")
                print(sampleembedding1)
                print("sample_emb2")
                print(sampleembedding2)                

                self.z_.data = torch.clamp(self.z_.data.detach().clone(), min=-Z_THRES, max=Z_THRES)

                image_tensors = self.G(self.z_, self.G.shared(self.y_))
                image_tensors = (image_tensors+1.) / 2.
                image_tensors = F.interpolate(image_tensors, size=224, mode=self.interp_mode)
                image_tensors.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
                
                logits_per_image, logits_per_text = self.clip_model(image_tensors, text_tok)
                logits_per_image = logits_per_image/clip_c
                if z_init is None:
                    z_init = self.z_.data.detach().clone()
                    y_init = self.y_.data.detach().clone()   #will change this one
                    score_init = logits_per_image.squeeze()
                    
                    #new commit to replace low score with nouns from the text to make faster iterations to reach the image
                    #only in first iteration
                    sorted, indices = torch.sort(score_init, descending=True)
                    y_init = y_init[indices]
                    countIter=len(endIndexNoun)
                    import random
                    y_init_counter=0 #for non -1 and non existent nouns
                    for iter in range (countIter):
                        if (endIndexNoun[iter]!=-1):
                            #case1:normal array [2,5] of 2 nouns found in imagenet
                            #start is 0 for first element else from previous element
                            #also works for noun doesn't exist in ImageNet in the middle so repetitive end as last time
                            #nouns:blue, photo(which is not found), dog : [2,2,5]
                            if (iter==0):
                                if(endIndexNoun[iter]==0):
                                    y_init[num_basis-1-y_init_counter]=nounsArrImageNet[0]
                                else: 
                                    y_init[num_basis-1-y_init_counter]=random.choice(nounsArrImageNet[0:endIndexNoun[iter]])
                                y_init_counter=y_init_counter+1
                                
                            #case2: not first element, and elemnts before it is normal different number(the current noun has classes in ImageNet)
                            #or: not first element, and elements before it is the same number(the current noun doesn't have class in ImageNet)
                            elif (iter!=0 and endIndexNoun[iter-1]!=-1 and endIndexNoun[iter-1]!=endIndexNoun[iter] ):
                                if(endIndexNoun[iter]==endIndexNoun[iter-1]+1):
                                    y_init[num_basis-1-y_init_counter]=nounsArrImageNet[endIndexNoun[iter]]
                                else:
                                    y_init[num_basis-1-y_init_counter]=random.choice(nounsArrImageNet[endIndexNoun[iter-1]+1:endIndexNoun[iter]])
                                y_init_counter=y_init_counter+1
                                
                            #case3: not the first element, and the noun doesn't exist in ImageNet in the beginning so -1, so begin at 0
                            elif (iter!=0 and endIndexNoun[iter-1]==-1):
                                if(endIndexNoun[iter]==0):
                                    y_init[num_basis-1-y_init_counter]=nounsArrImageNet[0]
                                else:
                                    y_init[num_basis-1-y_init_counter]=random.choice(nounsArrImageNet[0:endIndexNoun[iter]])
                                y_init_counter=y_init_counter+1


                else:
                    z_init = torch.cat([z_init, self.z_.data.detach().clone()], dim=0)
                    y_init = torch.cat([y_init, self.y_.data.detach().clone()], dim=0)
                    score_init = torch.cat([score_init, logits_per_image.squeeze()])

                sorted, indices = torch.sort(score_init, descending=True)
                z_init = z_init[indices]
                y_init = y_init[indices]
                score_init = score_init[indices]
                
                #print("z_init all")
                #print(z_init)
                print("y_init all")
                print(y_init)
                print("score_init all")
                print(score_init)
                
                z_init = z_init[:num_basis]
                y_init = y_init[:num_basis]
                score_init = score_init[:num_basis]
                
                #print("z_init")
                #print(z_init)
                print("y_init")
                print(y_init)
                print("score_init")
                print(score_init)
                
        
        #save_image(self.G(z_init, self.G.shared(y_init)), 'samples/init_%s.png'%text, 1)

        z_init_cllt.append(z_init.detach().clone())
        y_init_cllt.append(self.G.shared(y_init.detach().clone()))

        return z_init_cllt, y_init_cllt


    def optimize_clip_score(self, z_init_cllt, y_init_cllt, text, latent_noise=False, augment=True, opt_iters=500, optimize_y=False):

        text_tok = clip.tokenize([text]).to(self.device)
        clip_c = self.clip_model.logit_scale.exp()

        z_init_ans = torch.stack(z_init_cllt)
        y_init_ans = torch.stack(y_init_cllt)
        z_init_ans = z_init_ans.view(-1, z_init_ans.shape[-1])
        y_init_ans = y_init_ans.view(-1, y_init_ans.shape[-1])

        w_z = torch.randn((z_init_ans.shape[0], z_init_ans.shape[1])).to(self.device)
        w_y = torch.randn((y_init_ans.shape[0], y_init_ans.shape[1])).to(self.device)
        w_z.requires_grad = True
        w_y.requires_grad = True

        opt_y = torch.zeros(y_init_ans.shape).to(self.device)
        opt_y.data = y_init_ans.data.detach().clone()
        opt_z = torch.zeros(z_init_ans.shape).to(self.device)
        opt_z.data = z_init_ans.data.detach().clone()
        opt_z.requires_grad = True
        
        if not optimize_y:
            optimizer = torch.optim.Adam([w_z, w_y, opt_z], lr=5e-3, weight_decay=0.0)
        else:
            opt_y.requires_grad = True
            optimizer = torch.optim.Adam([w_z, w_y,opt_y,opt_z], lr=5e-3, weight_decay=0.0)

        for i in tqdm(range(opt_iters)):
            #print(w_z.shape, w_y.shape)
            optimizer.zero_grad()
            
            if not latent_noise:
                s_z = torch.softmax(w_z, dim=0)
                s_y = torch.softmax(w_y, dim=0)
                #print(s_z)
            
                cur_z = s_z * opt_z
                cur_y = s_y * opt_y
                cur_z = cur_z.sum(dim=0, keepdim=True)
                cur_y = cur_y.sum(dim=0, keepdim=True)

                image_tensors = self.G(cur_z, cur_y)
            else:
                s_z = torch.softmax(w_z, dim=0)
                s_y = torch.softmax(w_y, dim=0)
            
                cur_z = s_z * opt_z
                cur_y = s_y * opt_y
                cur_z = cur_z.sum(dim=0, keepdim=True)
                cur_y = cur_y.sum(dim=0, keepdim=True)
                cur_z_aug = cur_z + torch.randn(cur_z.shape).to(cur_z.device) * LATENT_NOISE
                cur_y_aug = cur_y + torch.randn(cur_y.shape).to(cur_y.device) * LATENT_NOISE
                
                image_tensors = self.G(cur_z_aug, cur_y_aug)
            
            loss = 0.0
            for j in range(image_tensors.shape[0]):
                if augment:
                    loss = loss + AugmentLoss(image_tensors[j:(j+1)], self.clip_model, text_tok, replicate=50, interp_mode=self.interp_mode)
                else:
                    loss = loss + NaiveSemanticLoss(image_tensors[j:(j+1)], self.clip_model, text_tok) 

            loss.backward()
            optimizer.step()

            opt_z.data = torch.clamp(opt_z.data.detach().clone(), min=-Z_THRES, max=Z_THRES)

        z_init_ans = cur_z.detach().clone()
        y_init_ans = cur_y.detach().clone()

        #save_image(self.G(z_init_ans, y_init_ans), 'samples/opt_%s.png'%text, 1)
        return self.G(z_init_ans, y_init_ans), z_init_ans, y_init_ans    

    def measureAugCLIP(self, z, y, text, augment=False, num_samples=20):
        text_tok = clip.tokenize([text]).to(self.device)
        avg_loss = 0.0
        for itr in range(num_samples):
            image_tensors = self.G(z, y)

            for j in range(image_tensors.shape[0]):
                if augment:
                    loss = AugmentLoss(image_tensors[j:(j+1)], self.clip_model, text_tok, replicate=50, interp_mode=self.interp_mode, policy=TEST_POLICY)
                else:
                    loss = NaiveSemanticLoss(image_tensors[j:(j+1)], self.clip_model, text_tok) 
            avg_loss += loss.item()

        avg_loss /= num_samples
        return avg_loss * (-1.)

