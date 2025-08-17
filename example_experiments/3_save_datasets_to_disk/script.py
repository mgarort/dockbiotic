from dockbiotic.data import disk


# pre-training datasets

disk.load_dockstring_data_from_disk()
disk.load_excape_data_from_disk()
disk.load_rdkit_data_from_disk()
disk.load_red_data_from_disk()

# fine-tuning datasets
disk.load_stokes_data_from_disk()
disk.load_coadd_data_from_disk(strain='atcc25922',
                               discard_above=0.9,
                               binarize=True)

# vs libraries
disk.load_chemdiv_data_from_disk()
