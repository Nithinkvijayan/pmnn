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
// Ultima alteracao: 2006 Nov 20
//

function [w,a,vet_erro,epoca] = redem_treinar(entrada,desejada,num_esp,eta,eta_gate,epocas_max,erro_max,deb)

// ------------------
// LIMPANDO VARIAVEIS
// ------------------

  x=0; y=0; p=0; q=0;
  qt_pontos=0; qt_pontosd=0;
  w=0; a=0; d=0;
  
// ------------------
// LEITURA DO ARQUIVO  
// ------------------
  
  // leitura dos dados de entrada
  x = fscanfMat(entrada);
  // leitura das respectivas respostas desejadas
  d = fscanfMat(desejada);
  
  // quant de pontos de treinamento e o tamanho de cada exemplo
  [qt_pontos, p] = size(x);
  
  // quant de pontos de treinamento deve ser igual a qt_pontos e tamanho de cada exemplo desejado
  [qt_pontosd, q] = size(d);

  // Verifica o tamanho dos arquivos
  if qt_pontos <> qt_pontosd
    disp('erro com os arquivos de treinamento: tamanho incompativel');
    abort
  end
  
// -------------
// INICIALIZACAO
// -------------

  // pesos dos modulos, matriz de 3 dimensoes
  w = zeros(p,q,num_esp);
  
  // pesos da rede de passagem, matriz de 2 dimensoes
  a = zeros(p,num_esp);
  
  // saidas dos modulos, matriz de 2 dimensoes
  fy = zeros(q,num_esp);
  
  // probabilidae a priori
  g = zeros(num_esp,1);
  u = zeros(num_esp,1);

  // probabilidade a posteriori
  h = zeros(num_esp,1);

  // valor do erro
  e = zeros(q,num_esp);

  // inicializacao dos pesos sinapticos com valores de -1 a 1
  w = (rand(w)*2) - 1;
//w = w + 0.25;
//disp(w,'w');
  
  // inicializacao dos pesos da rede de passagem com valores de -1 a 1
  a = (rand(a)*2) - 1;
//a = a + 0.25;
//disp(a,'a');
  
  // Indice para acesso ao vetor de entrada
  indice = [1:qt_pontos];
  
  vet_erro = zeros(epocas_max,1);
  erro_inst = zeros(qt_pontos,1);
  erro_medio = erro_max + 1;
  
// -------------------------------
// INICIO ALGORITMO DE TREINAMENTO
// -------------------------------

  // epocas de treinamento
  epoca = 0;

  // INICIO LACO DE TREINAMENTO
  // enquanto nao atingir o erro ou limite de epocas
  while ( (erro_max < erro_medio) & (epocas_max > epoca) )
  
    // condicao de parada por explosao do valor do erro
    if ( (erro_medio > 10^10 ) & ( epoca > 100 ) )
      break;
    end

    // zera o valor do erro medio quadrado
    erro_medio = 0;

    // erro em cada exemplo de uma epoca
    erro_inst = zeros(qt_pontos,1);

    // INICIO DE UMA EPOCA
    // para todos os exemplos de treinamento
    for n=1:qt_pontos
      
      // calculo dos u
      //--------------
      u = zeros(num_esp,1);
      for i=1:num_esp
        for xp=1:p
          u(i) = u(i) + a(xp,i) * x(indice(n),xp);
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
      fy = zeros(q,num_esp);
      for i=1:num_esp
        for m=1:q
          for xp=1:p
            fy(m,i) = fy(m,i) + ( x(indice(n),xp) * w(xp,m,i) );
          end
        end
      end
      
      // calculo dos erros
      //------------------
      e = zeros(q,num_esp);
      
      for i=1:num_esp
        for m=1:q
          e(m,i) = d(indice(n),m) - fy(m,i);
        end
      end

      // calculo das probabilidades a posteriori h
      //------------------------------------------
      hsoma = 0;
      hexp = zeros(num_esp,1);

      for i=1:num_esp
        temp = 0;
        for m=1:q
          temp = temp + e(m,i)*e(m,i);
        end
        hexp(i) = g(i) * exp( -0.5 * sqrt(temp) );
        hsoma = hsoma + hexp(i);
        erro_inst(n) = erro_inst(n) + sqrt(temp);
      end

      erro_inst(n) = erro_inst(n)/num_esp;
      
      for i=1:num_esp
        h(i) = hexp(i)/hsoma;
      end
      
      
      // atualizacao dos pesos sinapticos
      //---------------------------------

      for i=1:num_esp
        for m=1:q
          for xp=1:p
            w(xp,m,i) = w(xp,m,i) + (eta * h(i) * e(m,i) * x(indice(n),xp));
          end
        end
      end
      
      for i=1:num_esp
        for xp=1:p
          a(xp,i) = a(xp,i) + (eta_gate * ( h(i) - g(i) ) * x(indice(n),xp) );
        end
      end
      

      // DEBUG
      // -----
      if deb >= 2
        // epoca e exemplo
        printf("epoca %i - exemplo %i\n",epoca,n);

        // valor do u
        for i=1:num_esp
          printf("u(%i)=%.3f - ",i,u(i));
        end
        printf("\n");
        
        // valor de y de saída
        for i=1:num_esp
          printf("y(%i)=[ ",i);
          for m=1:q
            printf("%.3f ",fy(m,i));
          end
          printf("]\n");
        end
        
        // valor de g
        for i=1:num_esp
          printf("g(%i)=%.3f - ",i,g(i));
        end
        printf("\n");

        // h(i)
        for i=1:num_esp
          printf("h(%i)=%.3f - ",i,h(i));
        end
        printf("\n");
        
        // erro
        for i=1:num_esp
          printf("e(%i)=[ ",i);
          for m=1:q
            printf("%.3f ",e(m,i));
          end
          printf("]\n");
        end
          if deb == 3
            disp(w, 'w');
            disp(a, 'a');
          end
        lo=input("enter para continuar");
      end
    
    end // FIM DE UMA EPOCA
    
    erro_medio = (sum(erro_inst))/qt_pontos;

      // incrementa a epoca
    epoca = epoca + 1;

    vet_erro(epoca) = erro_medio;

    // alterar a ordem dos exemplos
//    indice = rand_indice(indice);


    if deb >= 1
      printf("epoca %i - erro_medio = %.8f\n",epoca,erro_medio);
    end
    
  end // FIM TREINAMENTO
  
  
endfunction



// //////////////////////////////////////////////////////////////
// Funcao para randomizar o indice de acesso aos dados de entrada
// //////////////////////////////////////////////////////////////
function [vet_indice] = rand_indice(vet_indice)

    // quantidade de pontos do indice
  qt_pontos = length(vet_indice)

    // cria vetor temporaria para a nova ordem do indice
  novo_ind = zeros(qt_pontos,1);
    
    // gera a ordem de troca dos dados
  for i=1:qt_pontos
      novo_ind(i) = floor(rand()*qt_pontos + 1);
  end

    // cria a nova ordem no indice vet_indice()
  for i=1:qt_pontos
      temp = vet_indice(novo_ind(i));
      vet_indice(novo_ind(i)) = vet_indice(i);
      vet_indice(i) = temp;
  end

endfunction


