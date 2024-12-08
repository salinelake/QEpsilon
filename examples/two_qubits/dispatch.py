import os


_sep = [1.26,1.43, 1.6, 1.68, 1.93, 2.35] # um
_atemp_ = [  100e-6 ] # K, note that 100uK temperature is at full tweezer depth, so actual thermal temp is 50uK?
_rtemp_ = [  25e-6 ] # K, note that 100uK temperature is at full tweezer depth, so actual thermal temp is 50uK?

os.system("echo '# batch submit' > batch_submit.sh")
## make a new folder and copy ./template.py to it, also replacing the placeholders
for sep in _sep:
    for atemp, rtemp in zip(_atemp_, _rtemp_):
        _data_file_ = 'data_XY8_{:.0f}'.format(sep*100)
        folder_name = 'sep{:.2f}um_atemp{:.0f}uK_rtemp{:.0f}uK'.format(sep, atemp*1e6, rtemp*1e6)
        os.makedirs(folder_name, exist_ok=True)
        os.system(f'cp ./template.py {folder_name}/run.py')
        os.system(f'sed -i "s/_data_file_/{_data_file_}/g" {folder_name}/run.py')
        os.system(f'sed -i "s/_sep_/{sep}/g" {folder_name}/run.py')
        os.system(f'sed -i "s/_atemp_/{atemp}/g" {folder_name}/run.py')
        os.system(f'sed -i "s/_rtemp_/{rtemp}/g" {folder_name}/run.py')
        os.system(f'cp ./submit.sh {folder_name}/submit.sh')
        os.system(f'echo "cd {folder_name}" >> batch_submit.sh')
        os.system(f'echo "sbatch submit.sh" >> batch_submit.sh')
        os.system(f'echo "cd .." >> batch_submit.sh')


