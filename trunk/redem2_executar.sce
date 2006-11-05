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
// Ultima alteracao: 2006 Out 27
//

// ////////////////////////////////////
// FUNCAO PARA O TREINAMENTO DA REDE M2
// ////////////////////////////////////

function [y] = redem2_executar(entrada,num_esp,arq_esp,w,func_esp,bias,arq_gate,a,func_gate,bias_gate)

// ------------------
// LIMPANDO VARIAVEIS
// ------------------

  x=0; y=0; p=0; q=0;
  qt_pontos=0; qt_pontosd=0;
  w=0; a=0; d=0;
  erro_inst=0; tmom=0; soma_grad=0;
  soma_erro=0;

// -------------  
// INICIALIZACAO
// -------------

    // sobre a arquitetura dos especialistas
  num_cam_esp = length(arq_esp);
  
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
  y = zeros(num_esp,1);
  
    // probabilidae a priori
  g = zeros(num_esp,1);
  

    
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


// ////////////////////////////////////
// Funcao de ativacao para os neuronios
// ////////////////////////////////////
function [y] = rna_funcao_ativacao( x, tipo )

    // Funcao linear
  if tipo == 0
  	y = x;
  end

   // Funcao Sigmoide
  if tipo == 1
  	y = 1/( 1 + exp(-x) );
  end

   // Funcao Tangente Hiperbolia
  if tipo == 2
  	a = 1.7159;
  	b = 2/3;
  	y = a*tanh(b*x);
  end

   // Funcao de limiar para 0.5
  if tipo == 3
  	if x < 0.5
  		y = 0;
  	else
  		y = 1;
  	end
  end

endfunction


