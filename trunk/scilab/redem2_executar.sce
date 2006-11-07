//
// UFRN - Universidade Federal do Rio Grande do Norte
// PPgEE - Programa de Pos-graduação em Engenharia Eletrica
//
// Rafael Marrocoso Magalhaes
//
// Codigo que implementa uma rede neural modular do tipo
// Modelo de Mistura Gaussiano Associativo com MLPs
// 
// Data criacao: 25 Out 2006
// Ultima alteracao: 2006 Nov 7
//

// ////////////////////////////////////
// FUNCAO PARA O TREINAMENTO DA REDE M2
// ////////////////////////////////////

function [y] = redem2_executar(entrada,num_esp,arq_esp,w,func_esp,bias,arq_gate,a,func_gate,bias_gate)

// ------------------
// LIMPANDO VARIAVEIS
// ------------------

  x=0; p=0; q=0;

// -------------  
// INICIALIZACAO
// -------------
  
    // leitura dos dados de entrada
  x = entrada

    // sobre a arquitetura dos especialistas
  num_cam_esp = length(arq_esp);

    // quant de pontos de treinamento e o tamanho de cada exemplo
  [p] = length(x);
 
    // quant de pontos de treinamento deve ser igual a qpontos e tamanho de cada exemplo desejado
  [q] = (arq_esp(num_cam_esp));

  
    // sobre a arquitetura da passagem
  num_cam_gate = length(arq_gate);
  
    // quantidade maxima de neurônios por camada esp
  max_neu_esp = max(arq_esp);

    // quantidade maxima de neurônios por camada de passagem
  max_neu_gate = max(arq_gate);

    // funcional dos neuronios, matriz de 3 dimensoes
    // neuronios, camadas, especialistas
  yi = zeros(max_neu_esp, num_cam_esp, num_esp);

  
    // funcional dos neuronios, matriz de 2 dimensoes
    // neuronios, camadas
  ai = zeros(max_neu_gate, num_cam_gate);

    // campo local induzido de todos os neuronios
    // neuronios camadas especialistas
    // campo da rede de passagem
  campo = zeros(max_neu_esp, num_cam_esp, num_esp);
  campo_gate = zeros(max_neu_gate, num_cam_gate);
  
    // armazena o valor da saída da rede
  y = zeros(q,1);
  
    // probabilidae a priori
  g = zeros(num_esp,1);
  

  soma_campo = 0;

// -------------------------------
// INICIO ALGORITMO DE TREINAMENTO
// -------------------------------

    
        // ---------------------
        // 1 - PASSO PARA FRENTE
        // ---------------------
        
        // para cada especialista
        // ----------------------
      for esp=1:num_esp
      
          // para cada camada dos especialistas
        for camada=1:num_cam_esp
        
            // caso especial de ser a primeira camada
          if camada == 1
            
              // para cada neuronio da camada de entrada
            for j=1:arq_esp(1)
              campo(j,camada,esp) = x(j);
              yi(j,camada,esp) = x(j);              
            end
            
          end // caso primeira camada
          
            // caso camada diferente da entrada
          if camada > 1
            
              // para cada neuronio da camada
            for j=1:arq_esp(camada)
              
              soma_campo = 0; // somatorio do campo local induzido
              
                // adicao do bias e seu peso
              soma_campo = soma_campo + ( w(1,j,(camada-1),esp) * bias );
              
                // atualizacao do campo devido neuronios anteriores
              for i=1:arq_esp(camada-1)
                soma_campo = soma_campo + ( w((i+1),j,(camada-1),esp) * yi(i,(camada-1),esp) );
              end
              
                // campo local induzido
              campo(j,camada,esp) = soma_campo;
              
                // saida do neuronio
              yi(j,camada,esp) = rna_funcao_ativacao( soma_campo, func_esp(camada) );
              
            end
          
          end // fim camada
          
        end // fim para camada espec
        
      end // fim para especialistas
      
        // para a rede de passagem
        // -----------------------        
      for camada=1:num_cam_gate
      
            // caso especial de ser a primeira camada
          if camada == 1
            
              // para cada neuronio da camada de entrada
            for j=1:arq_gate(1)
              campo_gate(j,camada) = x(j);
              ai(j,camada) = x(j);
            end
          
          end // caso primeira camada
          
            // caso camada diferente da entrada
          if camada > 1
            
              // para cada neuronio da camada
            for j=1:arq_gate(camada)
              
              soma_campo = 0; // somatorio do campo local induzido
              
                // adicao do bias e seu peso
              soma_campo = soma_campo + ( a(1,j,(camada-1)) * bias );
              
                // atualizacao do campo devido neuronios anteriores
              for i=1:arq_gate(camada-1)
                soma_campo = soma_campo + ( a((i+1),j,(camada-1)) * ai(i,(camada-1)) );
              end
              
                // campo local induzido
              campo_gate(j,camada) = soma_campo;
              
                // saida do neuronio
              ai(j,camada) = rna_funcao_ativacao( soma_campo, func_gate(camada) );
              
            end
          
          end // fim camada
          
      end // fim para rede de passagem

      
        // calculo das probabilidades a priori
        // -----------------------------------
      gsoma = 0; // somatorio das exp de gi
      gexp = zeros( num_esp, 1); // exp de gi
      
      for i=1:num_esp
        gexp(i) = exp(ai(i, num_cam_gate));
        gsoma = gsoma + gexp(i);
      end
      
      for i=1:num_esp
        g(i) = gexp(i)/gsoma;
      end
        
        // obtencao da saida da rede
        // -------------------------
      y = zeros(q,1);
      for i=1:q
        for esp=1:num_esp
          y(i) = y(i) + ( yi(i,num_cam_esp,esp) * g(esp) );
        end
      end
  
endfunction

