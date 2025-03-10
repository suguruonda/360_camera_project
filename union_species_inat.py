import glob
import os
import re
import difflib

ori_names = ["Papilio_Rutulus","Papilio_Zelicaon","Papilio_Indra","Papilio_Machaon","Parnassius_Smintheus",
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

ori_names2 = ["Danaus_Plexippus","Battus_Philenor","Hamadryas_Februa","Nymphalis_l-album","Speyeria_Callippe",
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

ori_names = ori_names + ori_names2

names = []
for i in ori_names:
    names.append(i.lower())


names = set(names)
folder_path = "/multiview/datasets/inat-butterflies-new/download_images/*"

folder_list = glob.glob(folder_path)
compare_list = []
for i in folder_list:
    compare_list.append(os.path.basename(i).lower())


match_list = []
for i in names:
    same = i in compare_list
    match_list.append(same)
    if same != True:
        temp = []
        for j in compare_list:
            r = difflib.SequenceMatcher(None, i, j).ratio()
            if r > 0.8:
                temp.append(j)
        if len(temp) > 0:
            print(i)
            print(temp)
#breakpoint()
for num,i in enumerate(names):
    if match_list[num] == False:
        print(i)

print(sum(match_list))

existing_list = ['Glaucopsyche_lygdamus', 'Colias_philodice', 'Aglais_milberti', 'Cercyonis_oetus', 'Speyeria_coronis', 'Junonia_coenia', 'Limenitis_weidemeyerii', 'Callophrys_gryneus', 'Nymphalis_californica', 'Parnassius_smintheus', 'Papilio_polyxenes', 'Anthocharis_julia', 'Erynnis_telemachus', 'Hesperia_juba', 'Pontia_beckerii', 'Chlosyne_leanira', 'Pieris_marginalis', 'Neophasia_menapia', 'Nathalis_iole', 'Cercyonis_meadii', 'Euchloe_lotta', 'Vanessa_atalanta', 'Papilio_rutulus', 'Chlosyne_acastus', 'Oeneis_jutta', 'Anthocharis_thoosa', 'Neominois_ridingsii', 'Oeneis_melissa', 'Polygonia_satyrus', 'Thorybes_pylades', 'Parnassius_clodius', 'Boloria_kriemhild', 'Papilio_indra', 'Epargyreus_clarus', 'Pieris_rapae', 'Papilio_machaon', 'Papilio_zelicaon', 'Hypaurotis_crysalus', 'Boloria_selene', 'Vanessa_cardui', 'Erebia_epipsodea', 'Danaus_gilippus', 'Colias_meadii', 'Amblyscirtes_vialis', 'Cercyonis_sthenele', 'Anthocharis_cethura', 'Colias_occidentalis', 'Oeneis_chryxus', 'Nymphalis_antiopa', 'Strymon_melinus', 'Colias_alexandra', 'Limenitis_archippus', 'Papilio_multicaudata', 'Callophrys_eryphon', 'Apodemia_mormo', 'Cyllopsis_pertepida', 'Colias_scudderi', 'Speyeria_cybele', 'Lycaena_hyllus', 'Pontia_occidentalis', 'Cupido_amyntula', 'Adelpha_eulalia', 'Papilio_eurymedon', 'Euchloe_ausonides', 'Pontia_protodice', 'Polites_sabuleti', 'Satyrium_behrii', 'Erebia_callias', 'Ochlodes_sylvanoides', 'Oeneis_uhleri', 'Cercyonis_pegala', 'Phyciodes_pulchella', 'Plebejus_melissa', 'Pontia_sisymbrii', 'Euphydryas_anicia', 'Celastrina_ladon', 'Colias_eurytheme', 'Satyrium_sylvinus', 'Phyciodes_cocyta', 'Oeneis_bore']


f = 'copy.sh'
with open(f, 'w', newline='\n') as file:
    file.write('#!/bin/bash\n')
    for num,i in enumerate(compare_list):
        if i in names:
            folder_path = folder_list[num]
            folder_name = os.path.basename(folder_path)
            if folder_name in existing_list:
                pass
            else:
                #file.write('mkdir ~/nobackup/archive/butterfly_inat/' + folder_name + '\n')
                file.write('scp ' + folder_path + '/adults/* ondas@orc:~/nobackup/archive/butterfly_inat/' + folder_name  + '\n')
            
            imglist = glob.glob(folder_path + '/adults/*.jpg')
            print(folder_name + " " + str(len(imglist)))

            #chamge this latter for too many files
            #find /multiview/datasets/inat-butterflies-new/download_images/Vanessa_atalanta/adults/ -type f -name "*" | xargs -I {} cp {} /home/suguru/sc/compute/butterfly_inat/Vanessa_atalanta