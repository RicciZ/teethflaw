import os, shutil

def filter(rootDir, desDir):
    if not os.path.exists(desDir):
        os.mkdir(desDir)
    ids = [11,12,13,14,15,16,17,21,22,23,24,25,26,27,28,31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]
    for i in ids:
        if not os.path.exists(os.path.join(desDir, str(i))):
            os.mkdir(os.path.join(desDir, str(i)))
    for root,dirs,files in os.walk(rootDir):
        id = os.path.split(root)[1]
        print(root)
        if id[0] == 'C':
            pid = id.split('_')[0]
            tid = os.path.split(os.path.split(root)[0])[1]
            save_path = os.path.join(desDir, tid, pid + '_' + tid)
            if not os.path.exists(save_path):
                shutil.copytree(root, save_path, symlinks=False, ignore=None)

holeDir = '/data/teethhole1221/bad'
saveDir = '/data/teethhole0101/bad'

filter(holeDir, saveDir)
