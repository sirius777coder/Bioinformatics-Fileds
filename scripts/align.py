import subprocess,re
def mmalign_wrapper(template, temp_pdbfile, force_alignment=None):
    if force_alignment == None:
        p = subprocess.Popen(f'/home/sirius/Desktop/MMalign {template} {temp_pdbfile} | grep -E "RMSD|TM-score=" ', stdout=subprocess.PIPE, shell=True)
    else:
        p = subprocess.Popen(f'/home/sirius/Desktop/MMalign {template} {temp_pdbfile} -I {force_alignment} | grep -E "RMSD|TM-score=" ', stdout=subprocess.PIPE, shell=True)
    output, __ = p.communicate()
    output = output.decode('utf-8')  # Decode the bytes to a string
    tm_rmsd  = float(str(output)[:-3].split("RMSD=")[-1].split(",")[0] )
    tm_score = float(str(output)[:-3].split("TM-score=")[-1].split("(normalized")[0] )
    aligned_length_line = [line for line in output.split('\n') if 'Aligned length=' in line][0]
    aligned_length = int(aligned_length_line.split("Aligned length=")[-1].split(",")[0].strip())
    seq_id = float(aligned_length_line.split("Seq_ID=n_identical/n_aligned=")[-1].split()[0].strip())
    return tm_rmsd, tm_score, aligned_length, seq_id


def tmalign_wrapper(template, temp_pdbfile, force_alignment=None):
    if force_alignment == None:
        p = subprocess.Popen(f'/home/sirius/Desktop/TMalign {template} {temp_pdbfile} | grep -E "RMSD|TM-score=" ', stdout=subprocess.PIPE, shell=True)
    else:
        p = subprocess.Popen(f'/home/sirius/Desktop/TMalign {template} {temp_pdbfile} -I {force_alignment} | grep -E "RMSD|TM-score=" ', stdout=subprocess.PIPE, shell=True)
    output, __ = p.communicate()
    output = output.decode('utf-8')  # Decode the bytes to a string
    tm_rmsd  = float(str(output)[:-3].split("RMSD=")[-1].split(",")[0] )
    tm_score = float(str(output)[:-3].split("TM-score=")[-1].split("(if")[0] )
    aligned_length_line = [line for line in output.split('\n') if 'Aligned length=' in line][0]
    aligned_length = int(aligned_length_line.split("Aligned length=")[-1].split(",")[0].strip())
    seq_id = float(aligned_length_line.split("Seq_ID=n_identical/n_aligned=")[-1].split()[0].strip())
    return tm_rmsd, tm_score, aligned_length, seq_id


def usalign_wrapper(template, temp_pdbfile, force_alignment=None):
    # -mm  1: alignment of two multi-chain oligomeric structures; 0 : (default) alignment of two monomeric structures
    # -ter 0:  align all chains from all models (recommended for aligning biological assemblies, i.e. biounits); 2 : (default) only align the first chain
    if force_alignment == None:
        p = subprocess.Popen(f'/home/sirius/Desktop/USalign {template} {temp_pdbfile} -mm 1 -ter 0 | grep -E "RMSD|TM-score=" ', stdout=subprocess.PIPE, shell=True)
    else:
        p = subprocess.Popen(f'/home/sirius/Desktop/USalign {template} {temp_pdbfile} -I {force_alignment} -mm 1 -ter 0| grep -E "RMSD|TM-score=" ', stdout=subprocess.PIPE, shell=True)
    output, __ = p.communicate()
    output = output.decode('utf-8')
    tm_rmsd = float(re.search('RMSD=\s*(\d+\.\d+)', output).group(1))
    tm_score = float(re.search('TM-score=\s*(\d+\.\d+)', output).group(1))
    return tm_rmsd, tm_score