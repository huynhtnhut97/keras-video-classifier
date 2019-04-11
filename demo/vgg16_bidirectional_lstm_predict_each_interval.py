import numpy as np
import sys
import os
import datetime

def main():
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

	from keras_video_classifier.library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier
	from keras_video_classifier.library.utility.ucf.UCF101_loader import load_ucf, scan_ucf_with_labels

	vgg16_include_top = True
	data_dir_path = os.path.join(os.path.dirname(__file__), 'very_large_data')
	model_dir_path = os.path.join(os.path.dirname(__file__), 'models', 'UCF-101')
	demo_dir_path = os.path.join(os.path.dirname(__file__), 'real-data')
	config_file_path = VGG16BidirectionalLSTMVideoClassifier.get_config_file_path(model_dir_path,
																				  vgg16_include_top=vgg16_include_top)
	weight_file_path = VGG16BidirectionalLSTMVideoClassifier.get_weight_file_path(model_dir_path,
																				  vgg16_include_top=vgg16_include_top)

	np.random.seed(42)

	load_ucf(data_dir_path)

	predictor = VGG16BidirectionalLSTMVideoClassifier()
	predictor.load_model(config_file_path, weight_file_path)

	print('reaching here three')
	
	unusual_actions = []
	usual_actions = []
	interval = 3
	txtFile = open('./reports/prediction_Each_Interval_Report'+'.txt','w')
	
	for video_file_path in os.listdir(demo_dir_path):
		if os.path.isfile(os.path.join(demo_dir_path,video_file_path)):
			print("Predicting video: {}".format(video_file_path))
			full_file_path = os.path.join(demo_dir_path,video_file_path)
			if(os.path.isfile(full_file_path)):
				predicted_labels = predictor.predict(full_file_path,interval=interval)
				print("TOTAL {} ACTIONS".format(len(predicted_labels)))
				for index in range(len(predicted_labels)):
					if(predicted_labels[index] == "Unusual action"):
						time = str(datetime.timedelta(seconds=index))
						print('predicted: ' + predicted_labels[index] + ' at: {0}'.format(time))
						unusual_actions.append(list((full_file_path,time)))
					elif(predicted_labels[index] == "Usual action"):
						time = str(datetime.timedelta(seconds=index))
						print('predicted: ' + predicted_labels[index] + ' at: {0}'.format(time))
						usual_actions.append(list((full_file_path,time)))
	print("TOTAL: {} unusual actions and {} usual actions".format(len(unusual_actions),len(usual_actions)))
	txtFile.write("Unsual actions")
	txtFile.write('\n')
	for item in unusual_actions:
		txtFile.write(item[0] + ' at '+ item[1])
		txtFile.write('\n')
	txtFile.write("Usual actions")
	txtFile.write('\n')
	for item in usual_actions:
		txtFile.write(item[0] + ' at '+ item[1])
		txtFile.write('\n')
if __name__ == '__main__':
	main()
