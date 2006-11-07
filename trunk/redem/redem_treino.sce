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

function [w,a,verro] = redem_treino(entrada,desejada,k,epocas,eta,eta2)


// LIMPANDO VARIAVEIS
//-------------------

  x=0; y=0; p=0; q=0;
  qpontos=0; qpontosd=0;
  w=0; a=0; d=0;
  

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
//  for xp=1:p
//    for m=1:q
//      for i=1:k
//        w(xp,m,i) = ( floor( rand()*50 + 1 ) )/100;
////        w(xp,m,i) = ( floor( (rand()*100 + 1 ) - 50) )/100;
//      end
//    end
//  end
  w = (rand(w)*2) - 1;
  
//  for xp=1:p
//      for i=1:k
//        a(xp,i) = ( floor( rand()*50 + 1 ) )/100;
////        a(xp,i) = ( floor( (rand()*100 + 1 ) - 50) )/100;
//      end
//  end
  a = (rand(a)*2) - 1;
  
  // Indice para acesso ao vetor de entrada
  vind = [1:qpontos];
  
  verro = zeros(epocas,1);
  errolocal = zeros(qpontos,1);
  
//  disp(size(w)); disp(w); disp(size(a)); disp(a);


// ALGORITMO
//----------

  // para todas as epocas
  for ep=1:epocas
    
     errolocal = zeros(qpontos,1);
  
    // para todos os exemplos de treinamento
    for n=1:qpontos
      
      // calculo dos u
      u = zeros(k,1);
      for i=1:k
        for xp=1:p
//          disp(a(xp,i)); disp(x(n,xp));
          u(i) = u(i) + a(xp,i) * x(vind(n),xp);
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
            yi(m,i) = yi(m,i) + ( x(vind(n),xp) * w(xp,m,i) );
          end
        end
      end
//      disp(size(w)); disp(w); disp(size(yi)); disp(yi);
      
      // calculo dos erros
      for i=1:k
        for m=1:q
          e(m,i) = d(vind(n),m) - yi(m,i);
        end
      end

      // calculo das probabilidades a posteriori h
      hsoma = 0;
//      errolocal(n) = 0;
      for i=1:k
        temp = 0;
        for m=1:q
          temp = temp + e(m,i)*e(m,i);
        end
//        hexp(i) = g(i) * exp( -0.5 * (temp^2) );
//        hexp(i) = g(i) * exp( -0.5 * (temp/2) );
//        hexp(i) = g(i) * exp( -0.5 * temp );
        hexp(i) = g(i) * exp( -0.5 * sqrt(temp) );
        errolocal(n) = errolocal(n) + sqrt(temp);
//        hexp(i) = g(i) * exp( -0.5 * (temp) );
        hsoma = hsoma + hexp(i);
//        printf("%f ",hsoma); 
      end
      errolocal(n) = errolocal(n)/k;
      
      for i=1:k
        h(i) = hexp(i)/hsoma;
      end
      
      
      // atualizacao dos pesos sinapticos
      for i=1:k
        for m=1:q
          for xp=1:p
            w(xp,m,i) = w(xp,m,i) + (eta * h(i) * e(m,i) * x(vind(n),xp));
          end
        end
      end
      
      for i=1:k
        for xp=1:p
          a(xp,i) = a(xp,i) + (eta2 * ( h(i) - g(i) ) * x(vind(n),xp) );
        end
      end
      
    end // fim dos exemplos
    
    verro(ep) = (sum(errolocal))/qpontos; 
    // alterar a ordem dos exemplos
    // cria vetor temporaria para a nova ordem do indice
    n_ordem = zeros(qpontos,1);
    
    // gera a ordem de troca dos dados
    for no=1:qpontos
      n_ordem(no) = floor(rand()*qpontos + 1);
    end
    
    // cria a nova ordem no indice vind()
    for ni=1:qpontos
      temp = vind(n_ordem(ni));
      vind(n_ordem(ni)) = vind(ni);
      vind(ni) = temp;
    end
    
    if( modulo(ep,(epocas/10)) == 0 )
      printf("%d ",ep);
    end
//    printf("epoca %d\n",ep);
    
  end // fim de epocas
  
  
endfunction
