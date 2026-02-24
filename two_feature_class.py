import pandas as pd
import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt

model_data = torch.load('model.pth')
w0 = model_data['weight'][0][0].item()
w1 = model_data['weight'][0][1].item()
b = model_data['bias'][0].item()

slope = -w0/w1
intercept = -b/w1

x1 = np.linspace(-1, 3, 100)
x2 = slope * x1 + intercept

line = pd.DataFrame({'x1': x1, 'x2': x2})

data = pd.read_csv('linearly_separable_2d.csv')


def make_plot():
	class_0 = data[data['y'] == 0]
	class_1 = data[data['y'] == 1]

	fig, ax = plt.subplots(figsize=(7, 5))
	ax.scatter(class_0['x1'], class_0['x2'], color='blue', s=35, alpha=0.8, label='Class 0')
	ax.scatter(class_1['x1'], class_1['x2'], color='red', s=35, alpha=0.8, label='Class 1')
	ax.plot(line['x1'], line['x2'], color='black', linewidth=2, label='Decision boundary')

	ax.set_xlabel('x1')
	ax.set_ylabel('x2')
	ax.set_title('Scatter Data with Model Line')
	ax.legend()
	ax.grid(alpha=0.25)
	fig.tight_layout()
	return fig


with gr.Blocks() as demo:
	gr.Plot(value=make_plot)

demo.launch()













