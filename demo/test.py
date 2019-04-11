import pickle as pk
with open("C:\\Users\\Admin\\Desktop\\Python\\Gits\\keras-video-classifier\\demo\\real-data\\HCVR_ch9_main_20181213110002_20181213111017.avi-VGG16-Features\\HCVR_ch9_main_20181213110002_20181213111017.pickle", 'rb') as pickle_file:
	content = pk.load(pickle_file)
	print(content)
