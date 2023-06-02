# Streamlit example for interacting with KneeFinder

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import glob
import pickle
from matplotlib.lines import Line2D

import pwlf

from knee_finder import KneeFinder



class StreamlitApp():
    def __init__(self):
        
        # Load initial data so relevant UI elements can be enabled
        self.sev_cap = self.load_severson_dict()
        self.sigmoid_array = self.load_fake_sigmoid()
        # Extract data from Severson cell 1 for initial plotting
        self.x, self.y = self.split_and_squeeze_array(self.sev_cap[1])
        self.have_data = True
        
        # Add the data source UI elements to the sidebar
        self.build_data_source_elements()
        
        # Assign values to self.x and self.y based on UI choices
        self.update_data_from_source()
        
        # Add the rest of the UI elements
        self.add_header_text()
        self.build_sidebar()
        

        # If we don't have data, we can't do analysis, so a blank section will show
        if self.have_data:
            self.perform_analysis()
            self.generate_plot()           
        else:
            st.markdown("**Please select a data source from the sidebar, or upload data in CSV format.**")
        
        
        self.add_footer_text()
        
        
    # Start of methods
    def add_header_text(self):
        st.header("KneeFinder")
        st.write("Original method developed by Paula Fermín-Cueto et al [1]. Method extended by Richard Gilchrist, Calum Strange, Goncalo dos Reis and Shawn Li [2]. Python implementation and web app by Richard Gilchrist.")
        st.write("[Source code](https://github.com/rg1990/knee-finder) on GitHub. [Accompanying article](https://www.google.com) published on Medium.")
        st.markdown("""---""")
        
        
    def add_footer_text(self):
        # Licensing stuff, source of data and credit to authors
        st.markdown("""---""")
        st.write("'Severson' data is described in [3] and can be downloaded [here](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204). All data is released under [CC BY 4](https://creativecommons.org/licenses/by/4.0/). ")
        st.write("[1] Paula Fermín-Cueto, Euan McTurk, Michael Allerhand, Encarni Medina-Lopez, Miguel F. Anjos, Joel Sylvester, Gonçalo dos Reis Identification and machine learning prediction of knee-point and knee-onset in capacity degradation curves of lithium-ion cells. Energy and AI, volume 1 (2020). https://doi.org/10.1016/j.egyai.2020.100006")
        st.write("[2] Strange C, Li S, Gilchrist R, dos Reis G. Elbows of Internal Resistance Rise Curves in Li-Ion Cells. Energies. 2021; 14(4):1206. https://doi.org/10.3390/en14041206")
        st.write("[3] Severson, K.A., Attia, P.M., Jin, N. et al. Data-driven prediction of battery cycle life before capacity degradation. Nat Energy 4, 383–391 (2019). https://doi.org/10.1038/s41560-019-0356-8")
        
        
        
    def build_data_source_elements(self):
        # File uploader located in sidebar
        self.uploaded_file = st.sidebar.file_uploader("Choose a CSV file to upload")

        # Sidebar for using own data or example data
        self.data_source_radio = st.sidebar.radio("Choose data source to display:",
                                                  ['Severson capacity (120 cells)',
                                                  'Sigmoid knee example (1 cell)',
                                                  'CSV Upload'])
        

        # Experiment with number input for cycling through Severson cells
        self.cell = st.sidebar.number_input("Choose a cell (Severson only)", min_value=1,
                                            max_value=120,
                                            step=1,
                                            value=1,
                                            disabled=not self.data_source_radio=="Severson capacity (120 cells)")
    
        
    def build_sidebar(self):
        
        # Checkbox for whether or not to scale y data to range [0, 1]
        # self.scale_y_checkbox = st.sidebar.checkbox("TODO - Fit using y /= max(y)?",
        #                                             disabled=True)
        

        # Sidebar piece with checkboxes to choose which curves to plot
        st.sidebar.markdown("**Choose curves to show on the plot:**")
        self.line_exp_checkbox = st.sidebar.checkbox("Line-Exponential", disabled=not self.have_data)
        self.bw_checkbox = st.sidebar.checkbox("Bacon-Watts", disabled=not self.have_data)
        self.bw2_checkbox = st.sidebar.checkbox("Double Bacon-Watts", disabled=not self.have_data)
        self.sigmoid_checkbox = st.sidebar.checkbox("Asymmetric Sigmoid (auto truncation only)", disabled=not self.have_data)


        # Sidebar radio buttons for truncation method
        self.truncation_radio = st.sidebar.radio("Choose truncation method:",
                                                ('None', 'Automatic', 'Manual'),
                                                disabled=not self.have_data)


        # Place truncation slider in sidebar. Disabled unless manual truncation selected
        if self.have_data:
            self.truncation_values = st.sidebar.slider('Select range of data to include:',
                                                      int(min(self.x)),  # lower limit
                                                      int(max(self.x)),  # upper limit
                                                      (int(min(self.x)), int(max(self.x))), # initial values
                                                      step=1,
                                                      disabled=self.truncation_radio!="Manual")


        # Checkboxes for comparison with other methods
        st.sidebar.markdown("""---""")
        st.sidebar.markdown("**Compare with other methods (point only)**:")
        self.pwlf_checkbox = st.sidebar.checkbox("Piecewise Linear", disabled=not self.have_data)
        
        # Create some space after the last UI element in the sidebar
        for i in range(5):
            st.sidebar.write("")


    def load_severson_dict(self):
        with open("severson_capacity.pkl", "rb") as a_file:
            cap_dict = pickle.load(a_file)
        return {i+1: cap_dict[cell] for i, cell in enumerate(list(cap_dict.keys()))}

    
    def load_fake_sigmoid(self):
        data = pd.read_csv("fake_double_sigmoid_data.csv", header=None, names=['x', 'y'])
        return data.to_numpy()
    
    
    def load_user_csv(self):
        if self.uploaded_file is not None:
            data = pd.read_csv(self.uploaded_file, header=None, names=['x', 'y'])
            return data.to_numpy()


    def update_data_from_source(self):
        '''
        Update the self.x and self.y arrays based on the current selection
        of data sources/cells from the UI elements
        '''
        self.have_data = False
        
        if self.data_source_radio == "CSV Upload":
            if self.uploaded_file is not None:
                user_arr = self.load_user_csv()
                self.x, self.y = self.split_and_squeeze_array(user_arr)
                self.have_data = True
            else:
                self.have_data = False
                
        elif self.data_source_radio == "Severson capacity (120 cells)":
            self.x, self.y = self.split_and_squeeze_array(self.sev_cap[self.cell])
            self.have_data = True
        
        elif self.data_source_radio == "Sigmoid knee example (1 cell)":
            self.x, self.y = self.split_and_squeeze_array(self.sigmoid_array)
            self.have_data = True
                

    def split_and_squeeze_array(self, arr):
        x, y = np.hsplit(arr, 2)
        return x.squeeze(), y.squeeze()


    def perform_analysis(self):
            
        if self.truncation_radio == "None":
            self.kf = KneeFinder(self.x, self.y, automatic_truncation=False)
        
        if self.truncation_radio == "Manual":
            self.kf = KneeFinder(self.x, self.y, automatic_truncation=False)
            trunc_lo, trunc_hi = self.truncation_values
            self.kf.update_manual_truncation(trunc_lo, trunc_hi)            
            self.kf.update_results()
            
        elif self.truncation_radio == "Automatic":
            self.kf = KneeFinder(self.x, self.y, automatic_truncation=True)
            self.kf.update_results()
        
        # Experimental - add PWLF if checkbox is checked
        if self.pwlf_checkbox:
            my_pwlf = pwlf.PiecewiseLinFit(self.x, self.y)
            self.breaks = my_pwlf.fit(2)

            self.x_hat = np.linspace(self.x.min(), self.x.max(), int(self.x.max()-self.x.min()+1))
            self.y_hat = my_pwlf.predict(self.x_hat)
            
             
    def generate_plot(self):
        
        fig, ax = plt.subplots()
        
        ax.plot(self.x, self.y, linewidth=4)
        data_line = Line2D(self.x,
                           self.y,
                           linestyle='-',
                           marker='o',
                           markersize=3,
                           linewidth=1,
                           label="Experimental Data")
        
        ax.add_line(data_line)
        #ax.lines = ax.lines[1:] # this is a hack to retain the scale obtained using plot
        _ = ax.lines.pop(0)
        
        if self.line_exp_checkbox==True:
            line_exp_line = Line2D(self.kf.x_trunc, self.kf.exp_fit, color='black', label='Line_Exp fit')
            ax.add_line(line_exp_line)
        if self.bw_checkbox==True:
            bw_line = Line2D(self.kf.x_trunc, self.kf.bw_fit, color='red', label='Bacon-Watts fit')
            ax.add_line(bw_line)
        if self.bw2_checkbox:
            bw2_line = Line2D(self.kf.x_trunc, self.kf.bw2_fit, color='orange', label='Double Bacon-Watts fit')
            ax.add_line(bw2_line)
            
        # Make sure the KneeFinder instance has a sig_fit curve that is not None
        if self.sigmoid_checkbox and self.kf.sig_fit is not None:
            sigmoid_line = Line2D(self.kf.x_cont, self.kf.sig_fit, color='purple', label='Sigmoid fit')
            ax.add_line(sigmoid_line)
        
        
        if self.pwlf_checkbox:
            ax.plot(self.x_hat, self.y_hat, color='turquoise', label='Piecewise Linear Fit')
            ax.axvline(self.breaks[1], color='green', label='Point (Piecewise Linear)')
        
        onset_line = Line2D((self.kf.onset, self.kf.onset), ax.get_ylim(), color='orange', label='Onset (KneeFinder)')
        point_line = Line2D((self.kf.point, self.kf.point), ax.get_ylim(), color='red', label='Point (KneeFinder)')
        ax.add_line(onset_line)
        ax.add_line(point_line)
        
        if self.kf.eol_cycle is not None:
            ax.axvline(self.kf.eol_cycle, color='black', label='EOL (80% initial)')
        
        # If truncation is enabled, add two vertical lines to indicate the truncation range
        if self.truncation_radio == "Manual":
            trunc_lo, trunc_hi = self.truncation_values
            ax.axvline(trunc_lo, linestyle='dashed', color='gray', alpha=1, label='Truncation limits')
            ax.axvline(trunc_hi, linestyle='dashed', color='gray', alpha=1)
            
            
        if self.truncation_radio == "Automatic":
            ax.axvline(self.kf.x_trunc[0], linestyle='dashed', color='gray', alpha=1, label='Truncation limits')
            ax.axvline(self.kf.x_trunc[-1], linestyle='dashed', color='gray', alpha=1) 
        
        # Plot labeling etc
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.4)
        #ax.autoscale(enable=False, axis='both')
        
        if self.data_source_radio == "Severson capacity (120 cells)":
            ax.set_title(f"Severson cell {int(self.cell)}")
        
        st.pyplot(fig)
        

app = StreamlitApp()

   