import os


_sep = [1.26,1.43, 1.6, 1.68, 1.93, 2.35] # um
# _sep = [1.26,1.43, 1.6, 1.68]
_atemp_ = [  18e-6 ] # K 
_rtemp_ = [  18e-6 ] # K 
# _tau_ = [ 2000, 4000, 6000, 8000] # us
_tau_ = [4000, 4000, 6000, 10000, 1000000, 1000000 ]
os.system("echo '# batch submit' > batch_submit.sh")
## make a new folder and copy ./template.py to it, also replacing the placeholders
for sep, tau in zip(_sep, _tau_):
    for atemp, rtemp in zip(_atemp_, _rtemp_):
        _data_file_ = 'data_XY8_{:.0f}'.format(sep*100)
        if tau > 1000:
            folder_name = 'atemp{:.0f}uK_rtemp{:.0f}uK_tau{:.0f}ms_sep{:.2f}um'.format(atemp*1e6, rtemp*1e6, tau*1e-3, sep)
        else:
            folder_name = 'atemp{:.0f}uK_rtemp{:.0f}uK_tau{:.0f}us_sep{:.2f}um'.format(atemp*1e6, rtemp*1e6, tau, sep)
        os.makedirs(folder_name, exist_ok=True)
        os.system(f'cp ./template.py {folder_name}/run.py')
        os.system(f'sed -i "s/_data_file_/{_data_file_}/g" {folder_name}/run.py')
        os.system(f'sed -i "s/_sep_/{sep}/g" {folder_name}/run.py')
        os.system(f'sed -i "s/_atemp_/{atemp}/g" {folder_name}/run.py')
        os.system(f'sed -i "s/_rtemp_/{rtemp}/g" {folder_name}/run.py')
        os.system(f'sed -i "s/_tau_/{tau}/g" {folder_name}/run.py')
        os.system(f'cp ./submit.sh {folder_name}/submit.sh')
        os.system(f'echo "cd {folder_name}" >> batch_submit.sh')
        os.system(f'echo "sbatch submit.sh" >> batch_submit.sh')
        os.system(f'echo "cd .." >> batch_submit.sh')


