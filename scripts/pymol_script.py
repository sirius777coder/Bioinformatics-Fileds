from pymol import cmd
import os
import os
import json
import numpy as np


# [mmalign(i,'TMP_029') for i in cmd.get_names("objects")]

def tmalign_folder_pdb(folder,save_path="./output.pse",methods="mmalign",align_pdb=None):
    cmd.reinitialize()
    cmd.do("run /home/sirius/Desktop/tmalign.py")
    for i in os.listdir(folder):
        if i.endswith(".pdb") or i.endswith(".cif"):
            object_name_without_extension = i.split(".")[:-1]
            object_name = ".".join(object_name_without_extension)
            cmd.load(f"{folder}/{i}",object_name)
    if align_pdb is not None:
        align_object = align_pdb.split("/")[-1]
        align_object = align_object.split(".")[:-1]
        align_object = ".".join(align_object)
        cmd.load(align_pdb,align_object)
    else:
        # random picke one protein in the folder as the align_object
        align_object = os.listdir(folder)[0]
        align_object = align_object.split(".")[:-1]
        align_object = ".".join(align_object)
    for obj in cmd.get_object_list():
        cmd.do(f"{methods} {obj}, {align_object}")
    cmd.set('grid_mode', 1)
    cmd.zoom()
    cmd.remove('solvent')
    cmd.save(save_path)


def motif_scaffolding_pymol_write(unique_designable_backbones,native_backbones,motif_json,save_path,native_motif_color="orange",design_motif_color="purple",design_scaffold_color="marine"):
    
    unique_designable_backbones_pdb = [i.replace(".pdb","") for i in os.listdir(unique_designable_backbones) if i.endswith('.pdb')]
    native_pdb = f"{native_backbones}/{unique_designable_backbones_pdb[0].split('_')[0]}.pdb"
    with open(motif_json,"r") as f:
        info = json.load(f)
    design_name_motif = {}
    for i in unique_designable_backbones_pdb:
        design_name_motif[i] = info[i]["motif_idx"]
    # re-initialize the pymol
    cmd.reinitialize()
    cmd.load(native_pdb, "native_pdb")
    contig = list(info.values())[0]["contig"]
    # "contig": "31-31/B25-46/32-32/A32/A4/A5"
    contig_list = [i for i in contig.split("/") if not i[0].isdigit()]
    config_folder = []
    for i in contig_list:
        chain = i[0]
        i = i[1:]
        if "-" in i:
            element = i.split("-")
            start = element[0]
            end = element[1]
            select = f"resi {start}-{end} and chain {chain}"
            config_folder.append(select)
        else:
            select = f"resi {i[1:]} and chain {chain}"
            config_folder.append(select)
    # merge all the contig into one
    config_extract = " or ".join(config_folder)
    print(f"loading native motif {config_extract}")

    cmd.extract("native_motif",config_extract)
    # delete native_pdb 
    cmd.delete("native_pdb")
    # color the native motif of PDB
    cmd.color(native_motif_color,"native_motif")
    cmd.show("sticks","native_motif")
    
    for i in os.listdir(unique_designable_backbones):
        print(i)
        if i.endswith(".pdb"):
            name = i.split(".")[0]
            cmd.load(f"{unique_designable_backbones}/{i}",name)
            cmd.color(design_scaffold_color,name)
            motif_residue = design_name_motif[name]
            cmd.select(f"{name}_motif","resi "+"+".join([str(i) for i in motif_residue])+" and "+name)
            cmd.color(design_motif_color,f"{name}_motif")
            cmd.show("sticks",f"{name}_motif")
            # align the motif
            cmd.align(f"{name}","native_motif")
    # set grid_mode to 1
    cmd.set("grid_mode",1)
    # zoom on the {name}
    cmd.zoom(f"{name}")
    cmd.save(save_path)