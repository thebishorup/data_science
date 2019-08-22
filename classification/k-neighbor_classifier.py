from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

classifier = KNeighborsClassifier()

classifier = classifier.fit(X, Y)

prediction = classifier.predict([[120, 56, 32]])
print(prediction)

acc = accuracy_score(['female'], prediction)
print(acc)
