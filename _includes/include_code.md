
<div class="tab code">
  <button class="tablinks python" onclick="openTab(event, 'python', 'interface')">
    <img src="/assets/images/Python.svg" alt="Python">
  </button>
  <button class="tablinks matlab" onclick="openTab(event, 'matlab', 'interface')">
    <img src="/assets/images/Matlab.png" alt="Matlab">
  </button>
<!--   <button class="tablinks" onclick="openTab(event, 'R', 'interface')">
    <img src="/assets/images/Rlogo.svg" alt="R">
  </button> -->
</div>
<div class="langcontent code interface python">
```python
#This code requires the G2S server to be running
{% assign url="https://raw.githubusercontent.com/GAIA-UNIL/G2S/master/example/python/" | append: include.exampleName | append: ".py" %}
{% remote_include url %}
```
</div>
<div class="langcontent code interface matlab">
```matlab
%This code requires the G2S server to be running
{% assign url="https://raw.githubusercontent.com/GAIA-UNIL/G2S/master/example/matlab/" | append: include.exampleName | append: ".m" %}
{% remote_include url %}
```
</div>