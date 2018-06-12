def init():
    """
    Init function of the scoring script
    """

    from sklearn.externals import joblib

    global reloaded_model

    # from azureml.assets.persistence.persistence import get_model_path
    # model_path = get_model_path('rf_attack_classifier_pipeline.pkl')
    # reloaded_model = joblib.load(model_path)
    reloaded_model = joblib.load('rf_attack_classifier.pkl')

def run(raw_data):

    import json

    try:
        phrase_list = json.loads(raw_data)['data']
        result = reloaded_model.predict(phrase_list)
    except Exception as e:
        result = str(e)
    return json.dumps({"result": result.tolist()})