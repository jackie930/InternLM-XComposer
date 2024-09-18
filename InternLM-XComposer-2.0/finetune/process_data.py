#process data
import json
import os

def update_x(x):
    x['image'] = os.path.join('data/task_images/train_imgs',x['video'].replace('.mp4','.jpg'))
    x = {k: v for k, v in x.items() if k != 'video'}
    x['conversations'][1]['from']='bot'
    return x

def update_x_2(x,type,folder_name):
    if type=='train':
        x['image'] = [os.path.join(f'data/{folder_name}/train_imgs',x['video'].replace('.mp4','.jpg'))]
    elif type=='test':
        x['image'] = [os.path.join(f'data/{folder_name}/test_imgs', x['video'].replace('.mp4', '.jpg'))]

    x = {k: v for k, v in x.items() if k != 'video'}
    x['conversations'][0]['value'] = x['conversations'][0]['value'].replace('<video>','<ImageHere>')
    x['conversations'][1]['from']='assistant'
    return x


def rewrite_file(input_json,version,type,folder_name):
    # Open the file and load the JSON data
    with open(input_json, 'r') as file:
        data = json.load(file)

    if version=='2.5':
        #update conversations
        data_intern = [update_x(i) for i in data]
    elif version=='2.0':
        data_intern = [update_x_2(i,type,folder_name) for i in data]

    #output data
    name = folder_name+'_'+type
    with open( f'data/{name}.json', 'w') as f:
        json.dump(data_intern, f)  # Optional parameter for indentation
    print('Data written to json')

if __name__ == '__main__':
    #rewrite_file('../../../../vila/data/1k_train.json')
    folder_name = 'task_images_25'
    #train
    rewrite_file('/home/ec2-user/SageMaker/vila/data/1k_train.json','2.0','train',folder_name)
    #test
    rewrite_file('/home/ec2-user/SageMaker/vila/data/1k_test.json', '2.0','test',folder_name)
