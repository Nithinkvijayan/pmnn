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
// Ultima alteracao: 2006 Out 18
//

function [y] = redem_run(x,w,a)


// INICIALIZACAO
//--------------

   k=0; p=0; q=0;

  // quant de pontos de treinamento e o tamanho de cada exemplo
  [p,q,k] = size(w);
  
  y = zeros(1,q);
  
  // saidas dos modulos, matriz de 2 dimensoes
  yi = zeros(q,k);
  
  // probabilidae a priori
  g = zeros(k,1);
  gexp = zeros(k,1);
  gsoma = 0;
  u = zeros(k,1);
  
// ALGORITMO
//----------

  // calculo dos u
  for i=1:k
    for xp=1:p
//          disp(a(xp,i)); disp(x(n,xp));
      u(i) = a(xp,i) * x(xp);
    end
  end
//      disp(size(u)); disp(u);
  
  // calculo das probabilidades a priori g
  gsoma = 0;
  for i=1:k
    gexp(i) = exp(u(i));
    gsoma = gsoma + gexp(i);
  end
  
  for i=1:k
    g(i) = gexp(i)/gsoma;
  end
  
  // calculo das saidas dos modulos
  yi = zeros(q,k);
  for i=1:k
    for m=1:q
      for xp=1:p
        yi(m,i) = yi(m,i) + ( x(xp) * w(xp,m,i) );
      end
    end
  end
//      disp(size(w)); disp(w); disp(size(yi)); disp(yi);
  
  // calculo dos erros
  for i=1:k
    for m=1:q
      y(m) = y(m) + (yi(m,i)*g(i));
    end
  end

  
endfunction
