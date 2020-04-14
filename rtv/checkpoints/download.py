import gdown
import subprocess

url = 'https://drive.google.com/uc?id=1e5c7308fdoCMRv0w-XduWqyjYPV4JWHS'

output = 'rtv.pth'
print('Donwloading {} ...'.format(output))
gdown.download(url, output, quiet=False)


# curl -L -o rtv.pth "https://drive.google.com/uc?export=download&id=1e5c7308fdoCMRv0w-XduWqyjYPV4JWHS"
