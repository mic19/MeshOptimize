# Optimize Mesh Blender Addon

This Addon helps detecting features on mesh and removing them. Detecting features was based on article 'Feature Detection for Surface Meshes' by Xiangmin Jiao and Michael T. Heath. 


<img src="/example_animation.gif" border="0" />

## Getting Started
This Addon uses operators from Addon 3D-Print Toolbox so it should be installed first. 
Script can be run two ways - as addon or standalone script.

In order to install as the addon in Blender download the project and move main project directory to your addons directory for Blender. Then go Edit > Preferences > Add-ons > Install. Find project directory and select *example.py*. Then enable it by checking the checkbox next to the addon name.

To run standalone script copy content of *optimize_giu.py* to Text Editor inside Blender and click Run Script.


## Usage
After selecting proper parameters for detecting features, click 'Select Features'. After some time (for complicated models), vertices that belong to a contour/border of the model (or a corner) are selected. 

Then it is time for user to correct the selection manually - for example some discontinuities may occur on borders. 

When all of the selection is correct, click 'Confirm' which will create Vertex Group for each feature. Now, they can be navigated by
'Previous' and 'Next' buttons and removed after clicking 'Optimize Features'.

Only small features like holes can be optimized - there is no optimization for borders but detected borders can be used in Decimate operator to assign weights.
