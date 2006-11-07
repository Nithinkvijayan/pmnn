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

function [w,a] = redem_treino(entrada,desejada,k,epocas,eta,eta2)


// LIMPANDO VARIAVEIS
//-------------------

  x=0; y=0; p=0; q=0;
  qpontos=0; qpontosd=0;
  w=0; a=0;
  

// LEITURA DO ARQUIVO  
//-------------------
  
  // leitura dos dados de entrada
  x = fscanfMat(entrada);
  // leitura das respectivas respostas desejadas
  d = fscanfMat(desejada);
  
  // quant de pontos de treinamento e o tamanho de cada exemplo
  [qpontos, p] = size(x);
//  disp('qpontos =' ,qpontos);
//  disp('p =' ,p);
  
  // quant de pontos de treinamento deve ser igual a qpontos e tamanho de cada exemplo desejado
  [qpontosd, q] = size(d);
//  disp(qpontosd,q);


  // Verifica o tamanho dos arquivos
  if qpontos <> qpontosd
    disp('erro com os arquivos de treinamento: tamanho incompativel');
    abort
  end
  
  
// INICIALIZACAO
//--------------

  // pesos dos modulos, matriz de 3 dimensoes
  w = zeros(p,q,k);
  
  // pesos da rede de passagem, matriz de 2 dimensoes
  a = zeros(p,k);
  
  // saidas dos modulos, matriz de 2 dimensoes
  yi = zeros(q,k);
  
  // probabilidae a priori
  g = zeros(k,1);
  gexp = zeros(k,1);
  gsoma = 0;
  u = zeros(k,1);
  
  // probabilidade a posteriori
  h = zeros(1,k);
  hexp = zeros(k,1);
  hsoma = 0;
  
  // valor do erro
  e = zeros(q,k);
  eexp = zeros(k,1);
  
  // inicializacao dos pesos sinapticos com valores de 0 a 0.5
  for xp=1:p
    for m=1:q
      for i=1:k
        w(xp,m,i) = ( floor( rand()*50 + 1 ) )/100;
//        w(xp,m,i) = ( floor( (rand()*100 + 1 ) - 50) )/100;
      end
    end
  end
  
  for xp=1:p
      for i=1:k
        a(xp,i) = ( floor( rand()*50 + 1 ) )/100;
//        a(xp,i) = ( floor( (rand()*100 + 1 ) - 50) )/100;
      end
  end
  
//  disp(size(w)); disp(w); disp(size(a)); disp(a);


// ALGORITMO
//----------

  // para todas as epocas
  for ep=1:epocas
  
    // para todos os exemplos de treinamento
    for n=1:qpontos
      
      // calculo dos u
      for i=1:k
        for xp=1:p
//          disp(a(xp,i)); disp(x(n,xp));
          u(i) = a(xp,i) * x(n,xp);
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
            yi(m,i) = yi(m,i) + ( x(n,xp) * w(xp,m,i) );
          end
        end
      end
//      disp(size(w)); disp(w); disp(size(yi)); disp(yi);
      
      // calculo dos erros
      for i=1:k
        for m=1:q
          e(m,i) = d(n,m) - yi(m,i);
        end
      end

      // calculo das probabilidades a posteriori h
      hsoma = 0;
      for i=1:k
        temp = 0;
        for m=1:q
          temp = temp + e(m,i);
        end
        hexp(i) = g(i) * exp( -0.5 * (temp^2) );
        hsoma = hsoma + hexp(i);
      end
      
      for i=1:k
        h(i) = hexp(i)/hsoma;
      end
      
      // atualizacao dos pesos sinapticos
      for i=1:k
        for m=1:q
          for xp=1:p
            w(xp,m,i) = w(xp,m,i) + (eta * h(i) * e(m,i) * x(n,xp));
          end
        end
      end
      
      for i=1:k
        for xp=1:p
          a(xp,i) = a(xp,i) + (eta2 * ( h(i) - g(i) ) * x(n,xp) );
        end
      end
      
    end // fim dos exemplos
    
  end // fim de epocas
  
  
endfunction
