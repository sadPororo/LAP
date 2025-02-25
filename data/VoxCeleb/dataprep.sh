# ## downloading speaker verification trials...
# wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt -P ./trials       # List of trial pairs - VoxCeleb1
# wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard.txt -P ./trials  # List of trial pairs - VoxCeleb1-H
# wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all.txt -P ./trials   # List of trial pairs - VoxCeleb1-E
# wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt -P ./trials      # List of trial pairs - VoxCeleb1 (cleaned)
# wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt -P ./trials # List of trial pairs - VoxCeleb1-H (cleaned)
# wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt -P ./trials  # List of trial pairs - VoxCeleb1-E (cleaned)

# check the arguments before you process below commands.
python dataprep.py -h

# ## downloading and processing VoxCeleb 1 & 2 datasets...
# python ./dataprep.py --save_path data --download --user USERNAME --password PASSWORD 
# python ./dataprep.py --save_path data --extract
# python ./dataprep.py --save_path data --convert --remove_aac --ncpu 12
# python ./dataprep.py --save_path data --preprocess
