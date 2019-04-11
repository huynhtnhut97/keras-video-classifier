import numpy as np
import sys
import os


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from keras_video_classifier.library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier
    from keras_video_classifier.library.utility.ucf.UCF101_loader import load_ucf, scan_ucf_with_labels

    vgg16_include_top = True
    data_dir_path = os.path.join(os.path.dirname(__file__), 'very_large_data')
    model_dir_path = os.path.join(os.path.dirname(__file__), 'models', 'UCF-101')
    demo_dir_path = os.path.join(os.path.dirname(__file__), 'bundle')
    config_file_path = VGG16BidirectionalLSTMVideoClassifier.get_config_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)
    weight_file_path = VGG16BidirectionalLSTMVideoClassifier.get_weight_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)

    np.random.seed(42)

    load_ucf(data_dir_path)

    predictor = VGG16BidirectionalLSTMVideoClassifier()
    predictor.load_model(config_file_path, weight_file_path)

    print('reaching here three')

    unusual_action_videos = []
    usual_action_videos = []
    txtFile = open('./reports/predictionsReport'+'.txt','w')
    for video_file_path in os.listdir(demo_dir_path):
        #label = videos[video_file_path]
        full_file_path = os.path.join(demo_dir_path,video_file_path)
        if(os.path.isfile(full_file_path)):
            predicted_label = predictor.predict(full_file_path)
            if(predicted_label == "Unusual action"):
                unusual_action_videos.append(full_file_path)
            elif(predicted_label == "Usual action"):
                usual_action_videos.append(full_file_path)
            print('predicted: ' + predicted_label)
    print("FOUND {} unusual actions".format(len(unusual_action_videos)))
    txtFile.write("Unusual actions")
    txtFile.write('\n')
    for item in unusual_action_videos:
        txtFile.write(item)
        txtFile.write('\n')
    txtFile.write("Usual actions")
    txtFile.write('\n')
    for item in usual_action_videos:
        txtFile.write(item)
        txtFile.write('\n')
        #correct_count = correct_count + 1 if label == predicted_label else correct_count
        #count += 1
        #accuracy = correct_count / count
        #print('accuracy: ', accuracy)


if __name__ == '__main__':
    main()
