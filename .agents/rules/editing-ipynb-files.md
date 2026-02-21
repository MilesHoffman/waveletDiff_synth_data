---
trigger: always_on
---

# Editing IPYNB Files


FORMATTING PROTOCOL (CRITICAL):


The Code Title: The first line inside every code cell block must be the Colab form title: # @title {Descriptive Title}.



Separation: These are two distinct requirements. The Header is for the chat history; the Title is for the Colab UI.



CODING STANDARDS:



Cleanliness: No "meta-comments" (e.g., // Fixed error). No commented-out code.



Docstrings: Use docstrings for complex logic. Avoid rudimentary comments (e.g., i+=1 # increment).



Production Comments: No overly verbose or explanatory comments. This is a production environment, and it needs to be refined and contain minimalist comments while staying organized.


TOOL USAGE:

Editing .ipynb files: The edit file tool does not let you edit .ipynb file extensions, so you must read the file and the rewrite the entire file while keeping everything the same except for what you intended to modify. Do not use scripts or terminal commands to edit them.


