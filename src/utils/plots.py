import matplotlib.pyplot as plt

def plot_predictions(train_data, train_labels, test_data, test_label, predictions=None):
  plt.figure(figsize=(10,7))
  #plotting training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  #plotting test data in green
  plt.scatter(test_data, test_label, c="g", s=4, label="Testing data")
  # Are there predictions?
  if predictions is not None:
    #plot the predictions if they exist
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
  #show legend
  plt.legend(prop={"size": 14})
  plt.xlabel("X")
  plt.ylabel("y")
  plt.show()

