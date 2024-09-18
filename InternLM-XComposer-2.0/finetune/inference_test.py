import torch
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm

torch.set_grad_enabled(False)

def load_model(model_nameorpath):
    # init model and tokenizer
    model = AutoModel.from_pretrained(model_nameorpath, trust_remote_code=True).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_nameorpath, trust_remote_code=True)
    return model, tokenizer

def single_infer(img_path,model,tokenizer):
    text = '<ImageHere> Is there any package theft behavior in the video? You need to determine whether the behavior is stealing a package, and answer "yes" or "no".'
    image = img_path
    with torch.cuda.amp.autocast():
      response, _ = model.chat(tokenizer, query=text, image=image, history=[], do_sample=False)

    return response

def main():
    model_nameorpath = 'output/finetune/checkpoint-156'
    model, tokenizer = load_model(model_nameorpath)
    '''
    img_path = 'data/images/Bk1Ghbw1SynAoDP7VS3B_1.jpg'
    res = single_infer(img_path,model,tokenizer)
    print (res)
    '''
    answers_file_eval = ("res.txt")
    ans_file_eval = open(answers_file_eval, "a")

    # Open the file and load the JSON data
    #todo: fix the image path in a_test does not exist
    with open('data/a_test.json', 'r') as file:
        data = json.load(file)

    for i in tqdm(data):
        test_image = i['image'][0]
        #print ("test image: ", test_image)
        gt = i['conversations'][1]['value']
        res = single_infer(test_image,model,tokenizer)

        ans_result = test_image + " " + res + " "+gt+" "
        ans_file_eval.write(ans_result + "\n")
        ans_file_eval.flush()

    #output_txt,
    with open('res.txt', "r") as f:
        lines = f.readlines()
    total_len = len(lines)
    label_total_yes = 0
    pred_total_yes = 0
    total_right = 0
    num = 0

    for i in lines:
        pred, label = i.split(' ')[1], i.split(' ')[2]
        if pred == label:
            total_right += 1
        if pred == 'Yes':
            pred_total_yes += 1
        if label == 'Yes':
            label_total_yes += 1
        if label == 'Yes' and pred == 'Yes':
            num += 1

    print("acc", total_right / total_len)
    print("recall: ", num / label_total_yes)
    print("precision: ", num / pred_total_yes)


if __name__ == '__main__':
    main()