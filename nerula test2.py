import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


weight_1_1 = 1.0
weight_2_1 = 1.0
weight_1_2 = 1.0
weight_2_2 = 1.0

bias_1=0
bias_2=0

def Classification(input1, input2, weight_1_1, weight_2_1, weight_1_2, weight_2_2, bias_1, bias_2):
    output1 = (input1 * weight_1_1) + (input2 * weight_2_1) + bias_1
    output2 = (input1 * weight_1_2) + (input2 * weight_2_2) + bias_2

    if output1 > output2:
        return 0
    else:
        return 1

def update(val):
    
    weight_1_1 = slider_weight_1_1.val
    weight_2_1 = slider_weight_2_1.val
    weight_1_2 = slider_weight_1_2.val
    weight_2_2 = slider_weight_2_2.val

    bias_1 = slider_bias_1.val
    bias_2 = slider_bias_2.val

    
    ax.clear()

    
    input1_values = np.linspace(0, 10, 100)
    input2_values = np.linspace(0, 10, 100)
    
    input1_grid, input2_grid = np.meshgrid(input1_values, input2_values)
    
    classification_results = np.array([Classification(x, y, weight_1_1, weight_2_1, weight_1_2, weight_2_2,bias_1,bias_2) 
                                       for x, y in zip(np.ravel(input1_grid), np.ravel(input2_grid))])
    classification_results = classification_results.reshape(input1_grid.shape)
    
    ax.contourf(input1_grid, input2_grid, classification_results, cmap='coolwarm', alpha=0.8)
    
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_title('Classification Results')
    plt.draw()


fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(left=0.25, bottom=0.35)


ax_slider_1 = plt.axes([0.25, 0.2, 0.65, 0.03])
ax_slider_2 = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_slider_3 = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_slider_4 = plt.axes([0.25, 0.05, 0.65, 0.03])

slider_weight_1_1 = Slider(ax_slider_1, 'SPIKE EFFECT ON P', -1, 1, valinit=0)
slider_weight_2_1 = Slider(ax_slider_2, 'SPOT EFFECT ON P', -1, 1, valinit=0)
slider_weight_1_2 = Slider(ax_slider_3, 'SPIKE EFFECT ON S', -1, 1, valinit=0)
slider_weight_2_2 = Slider(ax_slider_4, 'SPOT EFFECT ON P', -1, 1, valinit=0)


slider_weight_1_1.on_changed(update)
slider_weight_2_1.on_changed(update)
slider_weight_1_2.on_changed(update)
slider_weight_2_2.on_changed(update)

ax_slider_3 = plt.axes([0.25, 0.3, 0.65, 0.03])
ax_slider_4 = plt.axes([0.25, 0.25, 0.65, 0.03])


slider_bias_1 = Slider(ax_slider_3, 'bias1', -1, 1, valinit=0)
slider_bias_2 = Slider(ax_slider_4, 'bias2', -1, 1, valinit=0)


slider_bias_1.on_changed(update)
slider_bias_2.on_changed(update)



update(None)

plt.show()