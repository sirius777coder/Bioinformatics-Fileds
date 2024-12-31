import biotite.structure as struc
import biotite.structure.io as strucio

def convert_ptm_to_multimer(input,output,gap=200):
    atom_array = strucio.load_structure(input,model=1,extra_fields=["b_factor"])
    atom_list = []
    last_res_id = 0
    chain = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    chain_num = 0
    for atom in atom_array:
        # modify res id and chain id
        if atom.res_id - last_res_id -1 == gap:
            chain_num += 1
            modified_chain = chain[chain_num]
            res_id = atom.res_id - chain_num * gap 
        else:
            modified_chain = chain[chain_num]
            res_id = atom.res_id - chain_num * gap
        last_res_id = atom.res_id

        atom.res_id = res_id
        atom.chain_id = modified_chain
        atom_list.append(atom)

    array = struc.array(atom_list)

    # NEEDS TO BE MODIFEI
    # chain_set = list(set([i.chain_id for i in array]))
    # chain_leng_set = [len([i for i in array if i.chain_id == chain and i.atom_name =="CA"]) for chain in chain_set]

    # for atom in atom_list:
    #     atom_chain = chain_set.index(atom.chain_id)
    #     prev_aa = sum(chain_leng_set[:atom_chain])
    #     atom.res_id = atom.res_id - prev_aa
    # array = struc.array(atom_list)

    # save array 
    strucio.save_structure(output,array)

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_ptm_to_multimer(input_file,output_file)