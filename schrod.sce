dt = 0.001;
dx = .1;

Xmax = 100;
Xmin = -100;



//Add 
//-%i*dt*U(1,i)*U(1,i)*U(1,i)*eps
//to have NL terms
eps=-1;

function U=initPulse()
    U = Xmin:dx:Xmax;
    
    for i=1:length(U)
        U(i)=exp(-U(i)*U(i)/4)*exp(100*%i*U(i))
    end
    
endfunction



function R=doStep(U)
    M=Xmin:dx:Xmax;
	
	for i=1:length(M)
		if (i>1 & i<length(M)) then
			M(i)=U(2,i)-%i*dt/dx/2.*(U(1,i-1)+U(1,i+1)-2.*U(1,i));
		end
		if (i==1) then
			M(i)=U(2,i)-%i*dt/dx/2.*(U(1,i+1)-2.*U(1,i));
		end
		if(i==length(M)) then
			M(i)=U(2,i)-%i*dt/dx/2.*(U(1,i-1)-2.*U(1,i));
		end
	end
	
    R=[M;U];
endfunction

function R=firstStep(U)
    M=Xmin:dx:Xmax;
    
    //U(1,x)==temps t
    //U(2,x)==temps t-1
    
    for i=1:length(M)
		
		if (i==1) then
			M(i)=U(1,i)-%i*dt/dx/2.*(U(1,i+1)-2.*U(1,i));
		end
		if(i==length(M)) then
			M(i)=U(1,i)-%i*dt/dx/2.*(U(1,i-1)-2.*U(1,i));
		end
		if (i>1 & i<length(M)) then
			M(i)=U(1,i)-%i*dt/dx/2.*(U(1,i-1)+U(1,i+1)-2.*U(1,i));
		end
	end
	R=[M;U];
endfunction






function M=launch()
	Res=[];
U=initPulse();

U=firstStep(U);
Res=U;
for i=1:1000000	
	if(modulo(i,1000)==0)
		disp(i)
		Res=[U(1,:);Res];
	end
	
	if(i>3)
		U(3,:)=[];
	end
	U=doStep(U);
end
M=abs(Res);
endfunction





