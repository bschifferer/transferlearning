from sklearn.neighbors import NearestNeighbors

def get_classifier(hidden_features, k = 5):
    classifier = NearestNeighbors(n_neighbors = k)  
    classifier.fit(hidden_features)
    return(classifier)

def get_reco(classifier, item_set):
    reco = {}
    y_pred = classifier.kneighbors(item_set, return_distance=False)
    for i in range(0, y_pred.shape[0]):
        reco[i] = y_pred[i]
    return(reco)