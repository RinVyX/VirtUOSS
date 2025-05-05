# VirtUOSS

This is the first version of the script VirtUOSS.
It was made under the project <b> Well-e </b> , to normalize the videos of cows in their stall, hence normalizing the coordinates of the bounding boxes  

### How to use
<ol>
  <li>Choose the reference image, and select the four points of the stall</li>
  <li>Choose the target image/video (to transform), and select the four points</li>
  <li>Choose the croping (for the target image, have in mind the left/right interactions)</li>
  <li>Choose the path where to save the normalized video</li>
</ol>

### How does it work
#### Taking points
#### Optimized transformations values
using an objective function that try to optimize these transformations:
<ul>
  <li>Scaling (on x and y)</li>
  <li>Rotation</li>
  <li>Translation (on x and y)</li>
  <li>Skewing (on x and y)</li>
</ul>

<img url="/inetrpolation.png"></img>
to minimize the distance between the target points and the reference points.
