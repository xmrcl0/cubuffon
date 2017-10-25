Agulha de Buffon
================


Calcula o valor de PI usando a técnica de Buffon usando CUDA.


----

Compilação
----------

Efetuar a compilação do código-fonte:


        $ make 


----

Uso 
---
Executar com a opção help:


        $ ./cubuffon -h


----

Exemplos
--------

Lança uma agulha de tamanho 1 1000 vezes usando 1 bloco de tamanho 256: 


        $ buffon -n 1000 -l 1 -b 256 -x 1
