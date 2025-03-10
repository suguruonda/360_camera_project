import shutil
import glob
import os

sd_labels = ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']

###
names = ["Papilio_Rutulus","Papilio_Zelicaon","Papilio_Indra","Papilio_Machaon","Parnassius_Smintheus",
        "Pieris_Rapae","Pontia_Protodice","Pontia_Occidentalis","Neophasia_Menapia","Euchloe_Ausonides",
        "Anthocharis_Julia","Colias_Philodice","Colias_Eurytheme","Colias_Christina","Celastrina_Echo",
        "Plebejus_Melissa","Glaucopsyche_Lygdamus","Cupido_Amyntula","Icaricia_Lupini","Tharsalea_Rubidus",
        "Tharsalea_Heteronea","Lycaena_Hyllus","Tharsalea_Helloides","Polygonia_Satyrus","Nymphalis_Californica",
        "Hypaurotis_Crysalus","Strymon_Melinus","Satyrium_Sylvinus","Satyrium_Behrii","Callophrys_Gryneus",
        "Callophrys_Eryphon","Apodemia_Mormo","Nymphalis_Antiopa","Aglais_Milberti","Epargyreus_Clarus",
        "Thorybes_Pylades","Erynnis_Telemachus","Burnsius_Communis","Amblyscirtes_Vialis","Hesperia_Juba",
        "Ochlodes_Sylvanoides","Lon_Taxiles","Polites_Sabuleti","Vanessa_Atalanta","Vanessa_Cardui",
        "Junonia_Coenia","Chlosyne_Leanira","Chlosyne_Acastus","Euphydryas_Anicia","Phyciodes_Cocyta",
        "Phyciodes_Pulchella","Boloria_Kriemhild","Boloria_Selene","Adelpha_Eulalia","Limenitis_Weidemeyerii",
        "Limenitis_Archippus","Speyeria_Cybele","Speyeria_Coronis","Cercyonis_Pegala","Cercyonis_Pegala",
        "Papilio_Indra","Papilio_Zelicaon","Papilio_Polyxenes","Papilio_Machaon","Papilio_Machaon",
        "Papilio_Multicaudata","Papilio_Rutulus","Papilio_Eurymedon","Parnassius_Smintheus","Parnassius_Clodius",
        "Colias_Occidentalis","Colias_Eurytheme","Colias_Philodice","Colias_Alexandra","Colias_Meadii",
        "Colias_Scudderi","Nathalis_Iole","Neophasia_Menapia","Pontia_Occidentalis","Pontia_Protodice",
        "Pontia_Sisymbrii","Pontia_Beckerii","Pieris_Rapae","Pieris_Marginalis","Euchloe_Ausonides",
        "Euchloe_Lotta","Anthocharis_Julia","Anthocharis_Thoosa","Anthocharis_Cethura","Cercyonis_Sthenele",
        "Cercyonis_Oetus","Cercyonis_Meadii","Cercyonis_Pegala","Cercyonis_Pegala","Erebia_Epipsodea",
        "Erebia_Callias","Erebia_Magdalena","Coenonympha_California","Cyllopsis_Pertepida","Danaus_Gilippus",
        "Oeneis_Chryxus","Oeneis_Jutta","Oeneis_Bore","Oeneis_Melissa","Oeneis_Uhleri",
        "Neominois_Ridingsii"]

names2 = ["Danaus_Plexippus","Battus_Philenor","Hamadryas_Februa","Nymphalis_l_album","Speyeria_Callippe",
        "Polygonia_Interrogationis","Phoebis_Sennae","Vanessa_Virginiensis","Asterocampa_Celtis","Polygonia_Faunus",
        "Polygonia_Gracilis","Libytheana_Carinenta","Euptoieta_Claudia","Vanessa_Annabella",
        "Heliopetes_Ericetorum","Hylephila_Phyleus","Thymelicus_Lineola","Leptotes_Marina","Leptotes_Marina",
        "Echinargus_Isola","Brephidium_Exilis","Hemiargus_Ceraunus","Icaricia_Saepiolus","Celastrina_Echo",
        "Pholisora_Catullus","Hesperopsis_Libya","Polites_Themistocles","Glaucopsyche_Piasus","Zerene_Cesonia",
        "Abaeis_Nicippe","Erynnis_Horatius","Erynnis_Tristis","Erynnis_Propertius","Satyrium_Calanus",
        "Lerodea_Eufala","Callophrys_Augustinus","Satyrium_Titus","Icaricia_Lupini","Icaricia_Lupini",
        "Pyrisitia_Lisa","Chlosyne_Palla","Euphydryas_Editha","Phyciodes_Mylitta","Phyciodes_Tharos",
        "Euphyes_Vestris","Icaricia_Icarioides","Satyrium_Saepium","Atlides_Halesus","Erynnis_Icelus",
        "Satyrium_Californica","Hesperia_Pahaska","Speyeria_Egleis","Speyeria_Mormonia","Speyeria_Mormonia",
        "Megathymus_Yuccae","Hesperia_Leonardus","Chlosyne_Gorgone","Erynnis_Brizo","Lycaena_Cupreus",
        "Pyrgus_Ruralis","Hesperia_Colorado","Boloria_Chariclea","Chlosyne_Californica","Plebejus_Idas",
        "Plebejus_Idas","Callophrys_Spinetorum","Agriades_Glandon","Speyeria_Aphrodite","Speyeria_Aphrodite",
        "Speyeria_Coronis","Speyeria_Zerene","Speyeria_Hesperis","Speyeria_Hesperis","Argynnis_Nokomis",
        "Argynnis_Nokomis","Speyeria_Cybele","Speyeria_Cybele","Speyeria_Hydaspe","Boloria_Selene",
        "Abaeis_Mexicana","Junonia_Coenia","Papilio_Machaon","Papilio_Machaon","Burnsius_Communis",
        "Tharsalea_Editha","Lycaena_Arota","Cecropterus_Dobra","Tharsalea_Editha","Lycaena_Arota",
        "Lycaena_Nivalis","Lycaena_Hyllus","Lycaena_Nivalis","Lycaena_Hyllus","Lycaena_Dorcas",
        "Lycaena_Dorcas","Ochlodes_Yuma","Systasea_Zampa","Pyrgus_Scriptura","Poladryas_Arachne",
        "Phyciodes_Phaon","Piruna_Pirus","Limochores_Sonora","Chlorostrymon_Simaethis","Icaricia_Shasta",
        "Oarisma_Garita","Euphilotes_Enoptes","Callophrys_Sheridanii"]

names = names + names2

start_num = 211

num_copy_species = 3
###
for k in range(num_copy_species):
    species_number = start_num + k
    offset = 114 * (k)    
    arg = species_number -1
    if species_number == 60 or \
    species_number == 61 or \
    species_number == 62 or \
    species_number == 64 or \
    species_number == 67 or \
    species_number == 69 or \
    species_number == 72 or \
    species_number == 73 or \
    species_number == 78 or \
    species_number == 79 or \
    species_number == 80 or \
    species_number == 83 or \
    species_number == 85 or \
    species_number == 87 or \
    species_number == 125 or \
    species_number == 130 or \
    species_number == 144 or \
    species_number == 160 or \
    species_number == 171 or \
    species_number == 175 or \
    species_number == 176 or \
    species_number == 179 or \
    species_number == 181 or \
    species_number == 182 or \
    species_number == 185 or \
    species_number == 187 or \
    species_number == 190 or \
    species_number == 194 or \
    species_number == 195 or \
    species_number == 197 or \
    species_number == 198 or \
    species_number == 201:
        id = 2
    elif species_number == 65 or \
    species_number == 93 or \
    species_number == 183 or \
    species_number == 199 or \
    species_number == 145:
        id = 3
    elif species_number == 94 or \
    species_number == 188:
        id = 4
    elif species_number == 189:
        id = 5
    else:
        id = 1
    name = names[arg] + '-' + str(id).zfill(3)

    base_folder = "/data2/suguru/datasets/360camera/butterfly/original"
    dst_folder = base_folder + "/" + str(species_number).zfill(3) + "-" + name
    img_folder = dst_folder + "/images/" 
    cor_folder = dst_folder + "/scan_images/"
    print(k)
    print(str(species_number) + ' ' + name)
    try:
        os.makedirs(dst_folder)
    except FileExistsError:
        pass
    try:
        os.makedirs(img_folder)
    except FileExistsError:
        pass
    try:
        os.makedirs(cor_folder)
    except FileExistsError:
        pass

    for i in sd_labels: 
        files = glob.glob("/media/suguru/" + i + "/DCIM/*/*.JPG")
        print(i + " " + str(len(files)))
        files.sort()

        img_folder_c = img_folder + i +'/'
        try:
            os.makedirs(img_folder_c)
        except FileExistsError:
            pass

        suffix = ["","-i"]
        cor_b_name = ["x-","y-"]
        start = 2 + offset
        """
        if i == 'camera8':
            start = 1 + offset
        """
        count = 0
        for j in files[start:start + 36]:
            pos = 0
            cor_folder_num = cor_folder + str(pos) + '/'
            try:
                os.makedirs(cor_folder_num)
            except FileExistsError:
                pass
            cor_folder_num_c = cor_folder_num + i + '/'
            try:
                os.makedirs(cor_folder_num_c)
            except FileExistsError:
                pass

            shutil.copy(j, cor_folder_num_c + name \
                        + "-" + i + '-' \
                        + str(pos).zfill(2) + '-' \
                        + cor_b_name[(count)//18] \
                        + str((count)%18//2) + suffix[(count)%2] +".JPG")
            count += 1

        count = 0
        for j in files[start + 36 + 20:start + 36 + 20 + 36]:
            pos = 20
            cor_folder_num = cor_folder + str(pos) + '/'
            try:
                os.makedirs(cor_folder_num)
            except FileExistsError:
                pass
            cor_folder_num_c = cor_folder_num + i + '/'
            try:
                os.makedirs(cor_folder_num_c)
            except FileExistsError:
                pass

            shutil.copy(j, cor_folder_num_c + name \
                        + "-" + i + '-' \
                        + str(pos).zfill(2) + '-' \
                        + cor_b_name[(count)//18] \
                        + str((count)%18//2) + suffix[(count)%2] +".JPG")
            count += 1

        pos = 0
        for j in files[start + 36:start + 36 + 20]:
            shutil.copy(j, img_folder_c + name \
                        + "-" + i + '-' \
                        + str(pos).zfill(2) \
                        + ".JPG")
            pos += 1

        for j in files[start + 36 + 20 + 36:start + 36 + 20 + 36 + 20]:
            shutil.copy(j, img_folder_c + name \
                        + "-" + i + '-' \
                        + str(pos).zfill(2) \
                        + ".JPG")
            pos += 1

