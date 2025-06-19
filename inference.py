import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

def model_fn(model_dir):
    print(f"in model_fn Model directory is - {model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5
    model = models.efficientnet_b3(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Linear
                                     (num_features, 512),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(512, num_classes)
                                    )
    with open(os.path.join(model_dir, "model.pth"), "rb")as f:
        print("Loading Inventory Monitoring Model")
        checkpoint = torch.load(f, map_location = device)
        model.load_state_dict(checkpoint)
        print("Model Loaded")
        logger.info("model loaded successfully")
    model.eval()
    model.to(device)
    return model 
                            

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserialising the input data')
    logger.debug(f'Request body CONTENT-TYPE is : {content_type}')
    logger.debug(f'Request body TYPE is : {request_body}')

    if content_type == JPEG_CONTENT_TYPE:
        image = Image.open(io.BytesIO(request_body))
        logger.info('Loaded JPEG image')
        return image
    
    if content_type == JSON_CONTENT_TYPE:
        logger.debug(f'Request body is : {request_body}')
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON OBJECT: {request}')
        url=request.get('url')
        if not url:
            raise ValueError("URL not found in request body JSON")
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    raise Exception ('Requested unsupported ContentType in content_type: {}'.format(content_type))
def predict_fn(input_object, model):
    logger.info('Generating prediction based on input parameters.')

    test_transform = transforms.Compose([
        transforms.Resize(224),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  
                             [0.229, 0.224, 0.225]),
    ])
    logger.info('Transforming input')
    input_tensor = test_transform(input_object).unsqueeze(0)  # Add batch dimension

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    model.eval()

    with torch.no_grad():
        logger.info('Calling Model')
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)

        _, predicted_class_idx = torch.max(outputs, 1) 
        predicted_class_idx_scalar = predicted_class_idx.item()
        predicted_probabilities_list = probabilities.squeeze().tolist() 

        logger.info(f'Prediction made: class index {predicted_class_idx_scalar}, probabilities {predicted_probabilities_list}')

        return {
            "predicted_class_index": predicted_class_idx_scalar,
            "probabilities": predicted_probabilities_list}

def output_fn(prediction_output, accept='application/json'):
    logger.info('Serializing the generated output.')
    classes = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5'}

    predicted_class_index = prediction_output.get("predicted_class_index", -1)
    probabilities = prediction_output.get("probabilities", [])
    predicted_label = classes.get(predicted_class_index, "Unknown Class")

    response_body = {
        "predicted_class_index": predicted_class_index,
        "predicted_label": predicted_label,
        "probabilities": probabilities
    }

    json_output = json.dumps(response_body)
    logger.debug(f"Response body: {json_output}")

    if accept != 'application/json':
        raise Exception(f'Requested unsupported Accept header: {accept}. '
                        f'Currently only supports application/json')

    return json_output
  

                
                          

    
                 
    
    