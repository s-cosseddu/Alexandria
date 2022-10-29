# Alexandria

AI-powered tool to search your bookshelves.

[**Finalist at HackZurich 2021**](https://devpost.com/software/alexandria-0lkvrf), Alexandria was developed in collaboration with [notiv](https://github.com/notiv).


This repo contains the code of the original submission to HackZurich 2021, all project was completely designed and built in less than 24 hours.

More details at https://devpost.com/software/alexandria-0lkvrf

## Description

The web-app we built invites the user to upload pictures of bookshelves and search on the books for arbitrary strings. With the help of AI, the various objects are being located and text is being extracted. Finally, the location of the book with the corresponding string is being highlighted on the original image.

It is meant as a first step toward the creation of a mobile app that can detect text from a live stream and in real-time return locations of objects that match what the user is searching for. The final idea is to create a general search tool for labels on physical objects with a broad range of applications spanning from libraries to shops and warehouses.


## Technical details
We used a deep learning model (`YOLOv3`) with `opencv` and `tesseract` to perform object detection and text extraction. To get more meaningful results, we aligned the results with the `Google Books API`.

The web-app was built using `FastAPI`, based on the work and code described in the Medium article at [https://shinichiokada.medium.com/](https://shinichiokada.medium.com/) and [Building a Website Starter with FastAPI](https://levelup.gitconnected.com/building-a-website-starter-with-fastapi-92d077092864).
