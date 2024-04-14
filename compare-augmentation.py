import numpy as np 
import matplotlib.pyplot as plt 
 
# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 
 
# set height of bar
"""
dsName = 'DIABETES'
PREPROC = [.741, .732, .638, .611, .710, .662, .621, .721, .507, .599]
RAW = [.990, .776, .607, .709, .847, .689, .742, .740, .694, .704]

dsName = 'CANCER'
PREPROC = [.688, .678, .604, .527, .764, .795, .710, .671, .496, .441]
RAW = [1, .125, .500, .750, .312, .812, .500, .250, 1, .500]
"""
dsName = 'MALARIA'
PREPROC = [.627, .612, .591, .497, .673, .663, .635, .691, .500, .527]
RAW = [.818, .757, .576, .933, .781, .836, .614, .655, .819, .759]

# Set position of bar on X axis 
br1 = np.arange(len(PREPROC)) 
br2 = [x + barWidth for x in br1] 
 
# Make the plot
plt.bar(br1, PREPROC, color ='r', width = barWidth, 
        edgecolor ='grey', label = 'Augmentation') 
plt.bar(br2, RAW, color ='g', width = barWidth, 
        edgecolor ='grey', label ='No Augmentation') 
 
# Adding Xticks 
plt.ylabel('AUC', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(PREPROC))], 
        ['VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'DenseNet201', 'InceptionV3', 'Xception', 'MobileNetV2', 'EfficientNetV2B3', 'AlexNet'],
        rotation=45)
plt.title(f'AUC of the CNNs with the {dsName} datasets')
plt.legend()
plt.tight_layout()
#plt.show() 
plt.savefig(f'{dsName}-comparison.png')
plt.close()
