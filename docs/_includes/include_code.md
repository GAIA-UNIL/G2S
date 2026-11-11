
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
{% assign example_path = include.examplePath | default: include.exampleName %}
<div class="langcontent code interface python">
```python
#This code requires the G2S server to be running
{% assign local_path="example/python/" | append: example_path | append: ".py" %}
{% include_relative {{ local_path }} %}
```
</div>
<div class="langcontent code interface matlab">
{% if include.pythonOnly %}
<p><em>This example is only available in Python.</em></p>
{% else %}
```matlab
%This code requires the G2S server to be running
{% assign local_path="example/matlab/" | append: example_path | append: ".m" %}
{% include_relative {{ local_path }} %}
```
{% endif %}
</div>
