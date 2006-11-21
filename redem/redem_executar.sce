//
// UFRN - Universidade Federal do Rio Grande do Norte
// PPgEE - Programa de Pos-graduação em Engenharia Eletrica
//
// Rafael Marrocoso Magalhaes
//
// Codigo que implementa uma rede neural modular do tipo
// Modelo de Mistura Gaussiano Associativo.
// 
// Data criacao: 27 Set 2006
// Ultima alteracao: 2006 Nov 17
//

function [y] = redem_executar(x,w,a)

// ------------------
// INICIALIZACAO
// ------------------
   num_esp=0; p=0; q=0;

  // quant de pontos de treinamento e o tamanho de cada exemplo
  [p,q,num_esp] = size(w);
  
  y = zeros(1,q);
  
  // saidas dos modulos, matriz de 2 dimensoes
  fy = zeros(q,num_esp);
  
  // probabilidae a priori
  g = zeros(num_esp,1);
  u = zeros(num_esp,1);
  
// -------------------------------
// INICIO ALGORITMO DE TREINAMENTO
// -------------------------------

      // calculo dos u
      //--------------
      for i=1:num_esp
        for xp=1:p
          u(i) = u(i) + a(xp,i) * x(xp);
        end
      end
      
      // calculo das probabilidades a priori g
      //--------------------------------------
      gsoma = 0;
      gexp = zeros(num_esp,1);

      for i=1:num_esp
        gexp(i) = exp(u(i));
        gsoma = gsoma + gexp(i);
      end
      
      for i=1:num_esp
        g(i) = gexp(i)/gsoma;
      end
      
      // calculo das saidas dos modulos
      //-------------------------------
      for i=1:num_esp
        for m=1:q
          for xp=1:p
            fy(m,i) = fy(m,i) + ( x(xp) * w(xp,m,i) );
          end
        end
      end
        
      for i=1:num_esp
        for m=1:q
          y(m) = y(m) + (fy(m,i)*g(i));
        end
      end
  
  
endfunction

