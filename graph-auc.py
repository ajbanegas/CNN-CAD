# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

diseases = ("Diabetes", "Cancer", "Malaria")
auc_means = {
	'VGG16': (.741,.688,.627), 
	'VGG19': (.732,.678,.612), 
	'ResNet50': (.638,.604,.591), 
	'ResNet101': (.611,.527,.497), 
	'DenseNet': (.710,.764,.673), 
	'InceptionV3': (.662,.795,.663), 
	'Xception': (.621,.710,.635), 
	'MobileNet': (.721,.671,.691), 
	'EfficientNet': (.507,.496,.500), 
	'AlexNet': (.599,.441,.527)
}

x = np.arange(len(diseases))  # the label locations
width = 0.085  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in auc_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AUC')
ax.set_title('AUC of the CNN models by dataset')
ax.set_xticks(x + width, diseases)
ax.legend(loc='best', ncols=5, fontsize=10)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

