
<div class="tab code">
  <button class="tablinks python" onclick="openTab(event, 'python', 'interface')">
    <img src="{{ site.baseurl }}/assets/images/Python.svg" alt="Python">
  </button>
  <button class="tablinks matlab" onclick="openTab(event, 'matlab', 'interface')">
    <img src="{{ site.baseurl }}/assets/images/Matlab.png" alt="Matlab">
  </button>
<!--   <button class="tablinks" onclick="openTab(event, 'R', 'interface')">
    <img src="{{ site.baseurl }}/assets/images/Rlogo.svg" alt="R">
  </button> -->
</div>
<div class="langcontent code interface python">
```python
#This code requires the G2S server to be running
{% assign url="https://raw.githubusercontent.com/GAIA-UNIL/G2S/master/example/python/" | append: include.exampleName | append: ".py" %}
```
</div>
<div class="langcontent code interface matlab">
```matlab
%This code requires the G2S server to be running
{% assign url="https://raw.githubusercontent.com/GAIA-UNIL/G2S/master/example/matlab/" | append: include.exampleName | append: ".m" %}
```
</div>