import os
import subprocess

n = 1

with open('results.txt', 'w') as f:
    for test_dir in os.listdir('testes'):
        f.write(f'Teste {test_dir}:\n')
        for test in os.listdir(os.path.join('testes', test_dir)):
            f.write(f"\t{test}:\n")
            test_name = os.path.join('testes', test_dir, test)
            print(test_name)
            cuda = []
            cpp = []

            for i in range(n):
                sub = subprocess.Popen(f"misc/Config_CUDA.exe {test_name}", shell=True, stdout=subprocess.PIPE)
                subprocess_return = sub.stdout.read().decode("utf-8") 
                cuda.append(subprocess_return.strip())

                sub = subprocess.Popen(f"misc/Config_CPP.exe {test_name}", shell=True, stdout=subprocess.PIPE)
                subprocess_return = sub.stdout.read().decode("utf-8") 
                cpp.append(subprocess_return)

            list_join = ','.join(cuda)
            f.write(f"\t\tCUDA: {list_join}\n")
            list_join = ','.join(cpp)
            f.write(f"\t\tCPP: {list_join}\n")

        f.write('\n')
