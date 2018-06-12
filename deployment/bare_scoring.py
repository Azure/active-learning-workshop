def init():

    from sklearn.externals import joblib
    from azureml.assets.persistence.persistence import get_model_path

    global reloaded_model
    """
    Init function of the scoring script
    """
    model_path = get_model_path('rf_attack_classifier_pipeline.pkl')
    reloaded_model = joblib.load(model_path)

def run(raw_data):

    import json

    try:
        phrase_list = json.loads(raw_data)['data']
        result = reloaded_model.predict(phrase_list)
    except Exception as e:
        result = str(e)
    return json.dumps({"result": result.tolist()})