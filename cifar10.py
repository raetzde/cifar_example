import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from utils import visualize_images
# Test
# Pfad in welches das fertige Model gespeichert wird und von welchem ein trainiertes Model geladen wird
MODEL_PATH = './cifar_net_2epochs.pth'
# Auf False setzen, wenn man das Model nicht erneut trainieren will, sondern nur ein bestehendes Model testen will
TRAIN = True


# Die Funktion wird als erstes aufgerufen, wenn das Programm gestartet wird
def main():
    ########################################################################
    # Im folgenden wird das torchvision package genutzt, um den CIFAR10 Datensatz zu laden
    # Die Bilder in diesem Datensatz sind in einem Array gespeichert, welches die Werte im Intervall [0, 1] enthält.
    # (Eine 1 entspricht dann einem Farbwert von 255.)
    # Die folgende Transform-Funktion überträgt die Bilder in Tensoren, normalisiert diese und projeziert deren Werte
    # ins Intervall [-1, 1]. (Die Normalisierung dient dazu die Performance des Models zu verbessern.)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Hier werden der Trainings- und Testdatensatz geladen. Beim ersten Lauf des Programms werden diese automatisch
    # heruntergeladen und im data Ordner gespeichert. Der Trainingsdatensatz enthält 50.000 Bilder, der Testdatensatz
    # 10.000 weitere.
    # Mehr Änderungen von Giesau
    # Die Loader, die gebaut werden, dienen dazu die Iteration über die Bilder des Datensatzes zu vereinfachen. Sie
    # zerlegen den Datensatz in Batches (Gruppen) der angegebenen Größe und mischen diesen auch automatisch.

    # Wenn ein BrokenPipeError geworfen wird, dann einfach num_wokers auf 0 setzen.
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    batch_size = 8
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # die 10 Klassen, welche es zu klassifizieren gilt
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ########################################################################
    # Zeigt ein paar Bilder und die zugehörigen Label des Trainingdatensatzes an
    data_iter = iter(train_loader)
    images, labels = data_iter.__next__()
    # Zeigt die übergebenen Bilder mit echtem Label (oben)
    visualize_images(images, labels, classes)

    ########################################################################
    if TRAIN:
        # Das eigentliche Training
        net = train(train_loader)
    else:
        # Ansonsten wird einfach ein bestehendes Model geladen
        net = Net()
        net.load_state_dict(torch.load(MODEL_PATH))

    ########################################################################
    # Hier beginnt die Evaluierung des trainierten Models.
    # Die Test-Bilder werden dem Netzwerk übergeben, sodass dieses die Vorhersagen für die einzelnen Bilder trifft.
    data_iter = iter(test_loader)
    images, labels = data_iter.next()
    outputs = net(images)

    ########################################################################
    # Das Model produziert für jede Klasse ein Wahrscheinlichkeit. Hier wird der Index der Klasse ermittelt, für die
    # das Model die höchste Wahrscheinlichkeit berechnet hat.
    _, predicted_labels = torch.max(outputs, 1)

    # Visualisiert die Vorhersagen. Oben ist das echte Label das vorhergesagte Label befindet sich unter dem Bild.
    visualize_images(images, labels, classes, predicted_labels=predicted_labels)

    ########################################################################
    # Hier wird noch die Accuracy des Models berechnet bzw. welcher Prozentsatz der Bilder richtig Klassifiziert wurde.
    # Pures Raten würde bei 10 Klassen in einer Accuracy von 10% resultieren. Wenn das Model etwas gelernt hat, ist die
    # Accuracy deutlich höher.

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted_labels = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

    print(f'Genauigkeit des Netzwerks auf den 10.000 Bildern des Testdatensatzes: {100 * correct / total:.2f}%')

    ########################################################################
    # Hier noch eine genauere Aufschlüsselung der Accuracy für die einzelnen Klassen.

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted_labels = torch.max(outputs, 1)
            c = (predicted_labels == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f'Genauigkeit für die Klasse {classes[i]:5s} : {100 * class_correct[i] / class_total[i]:.2f}%')


# Diese Klasse enthält die Netzwerkdefinition des Models
class Net(nn.Module):
    # Constructor der die einzelnen Layer definiert
    def __init__(self):
        super(Net, self).__init__()
        # Dieser Layer ist ein spezieller Convolutional Layer. Diese werden oft in Verbindung mit Bildern verwendet, da
        # sie gut mit großen Mengen an Daten umgehen können, ohne dabei übermäßig viel Speicher zu verbrauchen.
        # Dabei werden kleine Matrizen (sogenannte Filter oder Kernel) über die Eingabe geschoben. Zwischen dem Filter
        # und dem überlagerten Ausschnitt der Eingabedaten wird dann das inner Produkt berechnet. Die Grafik auf der 
        # Wikipedia-Seite stellt das ganz gut dar: https://de.wikipedia.org/wiki/Convolutional_Neural_Network
        self.conv1 = nn.Conv2d(3, 6, 5)

        # Ein Maximum Pooling Layer. Ist sehr eng verwandt mit dem Convolutional Layer. Wird auch auf der
        # Wikipedia-Seite erklärt.
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # fc = Kurform von Fully-Connected
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Die Softmax-Funktion verwandelt die Ausgabe des Netzwerks in eine Wahrscheinlichkeitsverteilung. Die 10
        # Werte der Neuronen des letzten Layers werden dann ins Intervall [0, 1] projeziert, sodass diese Werte zusammen
        # 1 bzw. 100% ergeben.
        self.softmax = nn.Softmax(dim=1)

    # Diese Funktion definiert (wie der Name sagt) den Forward-Pass.
    def forward(self, x):
        # Im ersten Block werden die Eingaben erst in den Convolutional-Layer, dann in die ReLU-Aktivierungsfunktion und
        # schließlich in den max-pooling Layer gegeben.
        x = self.pool(F.relu(self.conv1(x)))
        # Analog hier
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # verändert lediglich die Dimensionen des Tensors
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def train(train_loader):
    ########################################################################
    net = Net()

    # In der criterion Variable wird unsere Loss-Funktion gespeichert. Das ist in diesem Fall Cross-Entropy Loss,
    # welches die am häufigsten genutzte Loss-Funktion für Klassifikationsprobleme ist.
    criterion = nn.CrossEntropyLoss()

    # Der optimizer nutzt Stochastic Gradient Descent als Algorithmus. Die lr (learing rate) gibt an, wie stark sich die
    # Gewichte bei jedem Update verändern bzw. wie "groß die Schritte sind", die der Algorithmus entlang der
    # Loss-Funktion in Richtung Minimum geht. (vgl. https://cdn-images-1.medium.com/max/600/1*iNPHcCxIvcm7RwkRaMTx1g.jpeg)
    # Momentum ist ein weiterer Parameter zur Konfiguration dieses Algorithmus.
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    ########################################################################
    # Ab hier startet das Training. In jeder Epoche wird ein Mal der gesamte Datensatz durch das Netzwerk gegeben
    num_epochs = 2
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Der train_loader gibt bei jeder Iteration ein Batch der Größe 4 (=4 Bilder + zugehörige Labels) zurück, die in
        # die data Variable geschrieben werden.
        for i, data in enumerate(train_loader, 0):
            # In inputs sind jetzt die 4 Bilder gespeichert.
            inputs, labels = data

            # Obligatorischer Schritt vor jedem Update der Parameter (=weights + biases)
            optimizer.zero_grad()

            outputs = net(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Loss-Berechnung
            loss.backward()  # Backward-Pass/Gradient-Berechnung
            optimizer.step()  # Parameter-Update

            # Zeigt den Loss alle 2000 Batches an
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Training beendet')

    ########################################################################
    # Das trainierte Model wird gespeichert, damit es für zukünftige Verwendung nicht neu trainiert werden muss.
    torch.save(net.state_dict(), MODEL_PATH)
    return net


if __name__ == '__main__':
    main()
