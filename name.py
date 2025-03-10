import shutil
import glob
import os

names = ["Papilio_Rutulus","Papilio_Zelicaon","Papilio_Indra","Papilio_Machaon","Parnassius_Smintheus",
        "Pieris_Rapae","Pontia_Protodice","Pontia_Occidentalis","Neophasia_Menapia","Euchloe_Ausonides",
        "Anthocharis_Julia","Colias_Philodice","Colias_Eurytheme","Colias_Christina","Celestrina_Ladon",
        "Plebejus_Melissa","Glaucopsyche_Lygdamus","Cupido_Amyntula","Plebejus_Acmon","Tharsalea_Rubidus",
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

names2 = ["Danaus_Plexippus","Battus_Philenor","Hamadryas_Februa","Nymphalis_l-album"]
start_num = 1

f = 'butterfly/*'
f_list = glob.glob(f)
f_list.sort()
###
for count, ele in enumerate(f_list):
    species_number = start_num + count
    arg = count
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
    species_number == 87:
        id = 2
    elif species_number == 65 or \
    species_number == 93:
        id = 3
    elif species_number == 94:
        id = 4
    else:
        id = 1
    name = "butterfly/" +str(species_number).zfill(3) + "-" + names[arg] + '-' + str(id).zfill(3) + ".JPG"

    os.rename(ele, name)