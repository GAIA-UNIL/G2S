---
title: Algorithms
author:
  - Mathieu Gravey
date: 2023-03-13
toc-depth: 2
---

To use the GeoStatistical Server (G2S), you need to install both the G2S server and an interface that matches your preferred programming language. The server handles the core functionality and computations, while the interface enables communication between your code and the server. Follow the steps below to install both components:

1. **Install the G2S server**: The server is the backbone of the G2S framework, responsible for running the simulations. To install the G2S server, follow the instructions provided in the [server installation guide](installation/server.html).

2. **Install the G2S interface**: Choose the interface that corresponds to your preferred programming language (Python, MATLAB, or R). These interfaces are written in C++, ensuring seamless integration and performance. To install the appropriate G2S interface, refer to the instructions in the [interface installation guide](installation/interfaces.html).

After completing the installation of both the G2S server and the corresponding interface, you can start running stochastic simulations using the G2S framework in your preferred programming language.

## Explanation

The GeoStatistical Server (G2S) utilizes a server-interface architecture to provide a versatile, efficient, and user-friendly platform for running geostatistical simulations. This design choice offers several key advantages:

- Language and code-independence: By decoupling the server and algorithm implementation (written in highly optimized C++) from the interface, G2S interface can work seamlessly with various programming languages, including Python, MATLAB, and R. Users can run simulations using their preferred language without worrying about compatibility issues.

- Flexibility and extensibility: The server-interface architecture allows for easy integration of new algorithms and techniques as they become available. This modular design ensures that the framework can be updated and improved without disrupting existing workflows.

- Scalability and performance: The algorithms are written in C++, a language known for its high performance and efficiency, ensuring that the simulations are executed quickly and reliably. Additionally, the server can be set up to handle multiple simultaneous requests and remotely, making it suitable for large-scale applications.

The G2S framework supports Python, MATLAB, and R interfaces, ensuring seamless integration with popular scientific programming languages while maintaining the performance benefits of the underlying C++ implementation. A Python server interface (to implement algorithms in Python) is also available. While originally designed for prototyping, it allows for a quick conversion of existing workflows in G2S.
