package de.mdelab.morisia.comparch.simulator.impl;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import de.mdelab.morisia.comparch.ArchitecturalElement;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.Injection;
import de.mdelab.morisia.comparch.simulator.InjectionStrategy;
import de.mdelab.morisia.comparch.simulator.IssueType;

public class ValidationInjectionStrategy implements InjectionStrategy {

	private IssueType[] issueTypes;
	private Architecture eArchitecture;
	
	public ValidationInjectionStrategy(IssueType[] issueTypes,
			Architecture eArchitecture) {
		this.issueTypes = issueTypes;
		System.out
		.println("Validation injection strategy for experiments may take the following issues into account: "
				+ Arrays.toString(this.issueTypes));
		this.eArchitecture = eArchitecture;
		

	}

@Override
public List<Injection<? extends ArchitecturalElement>> getInjections(int runCount) {

	List<Injection<? extends ArchitecturalElement>> injections = new LinkedList<Injection<? extends ArchitecturalElement>>();

	//if(runCount==50) 
		//if(runCount==25)
			//if(runCount==50)
	if(runCount==5||runCount==25||runCount==50||
			  runCount==56||runCount==76||runCount==101||
			  runCount==107||runCount==127||runCount==152||
			  runCount==158||runCount==178||runCount==203||
			  runCount==209||runCount==229||runCount==254||
			  runCount==260||runCount==280||runCount==305)
			
			{	int run= Math.floorMod(runCount,51);
			 System.out.println("\n run =  CF divide "  + run);
				 //System.out.println("\n ============================================================================= CF size  " +run);
				int startingPoint=Math.floorDiv(runCount,50);
				// System.out.println("\n shop starts at  " +startingPoint);
				int shop=run+startingPoint;
				int cfType=3;
				int extra=Math.floorMod(run, cfType);
				 //System.out.println(" extra =" + extra);
				int interval=Math.floorDiv(run, cfType);
				// System.out.println("interval =" + interval);
			    int  f1=interval;
			     int f2=interval;
				
				int evalRound=3;// t=x;
				int comp=shop+evalRound;
				int c1=0;
				int c2 =0;
				int c3=0;
				
			int f3=interval+extra;
			//	int cMax=18;
			  int cMax=17;
			 int cInd=cMax-1;
			
				
			/*for ( shop=0;shop<100; shop++)
			{
				for ( comp=0;comp<18;comp++)

					if (this.getComponent(shop, comp).getPerformance()==0)
					{System.out.println("\n shop:  "+ shop+"   Comp:  "+comp);}
					
				
			}*/
			
				while(c1<f1)
				{	c1++;
					if(shop>99)
					shop = Math.floorMod(shop, 100);
					 comp++;
						if(comp>cInd)
							comp = Math.floorMod(comp, cMax);
						injections.add(new Injection<Component>(IssueType.CF1, this
								.getComponent(shop, comp)));
						// System.out.println("\n CF1, " + this.getComponent(shop, comp).getType().getName());
						 // System.out.println("\n CF1  shop:  "+ shop + "  Comp:  "+comp+"\n");
					shop++;
				}	
				// System.out.println(" ___________________________________         f1 "+ c1 );

				while(c2<f2)
				{	c2++;
					if(shop>99)
					shop = Math.floorMod(shop, 100);
					 comp++;
						if(comp>cInd)
							comp = Math.floorMod(comp, cMax);
						injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
								.getComponent(shop, comp).getProvidedInterfaces().get(0)));
						//  System.out.println("\n CF2  shop:  "+ shop + "  Comp:  "+comp+"\n");
					
					// System.out.println("\n CF2, " + this.getComponent(shop, comp).getType().getName());
					 shop++;
				}
				// System.out.println(" ___________________________________         f2 "+ c2);

				while(c3<f3)
				{	c3++;
					if(shop>99)
					shop = Math.floorMod(shop, 100);
					 comp++;
						if(comp>cInd)
							comp = Math.floorMod(comp, cMax);
						injections.add(new Injection<Component>(IssueType.CF3, this
								.getComponent(shop, comp)));
						// System.out.println("\n CF3  shop:  "+ shop + "  Comp:  "+comp+"\n");
					
					// System.out.println("\n CF3, " + this.getComponent(shop, comp).getType().getName());
					 shop++;
				}
				// System.out.println(" ___________________________________         f3 "+ c3);
		
				// System.out.println(" ___________________________________         f5 "+ c5);	

			}
			
	
	// System.out.println(" ___________________________________         f3 "+ c3);

	//////////////////////////////////////////////////////////////////////////////////////////
	/*			
	 if (runCount == 1) {
		 //1 CF
		 System.out.print("\n 1st mape injects 1 CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(1, 1)));
		
		
	
		
	} else if (runCount == 2) {
		//5 CF
		System.out.print("\n2nd mape injects 5 more CF");
		this.updateReliabiliy_FB_2();
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getAuthenticationComponent(1)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(2, 5)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(3, 7)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(4, 11)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(5, 8)));
	
		
	} else if (runCount == 3) {
		//10 CF
		System.out.print("\n 3rd mape injects 10 more CF");
		
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(6, 4)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(7, 9)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(8, 10)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(8, 2)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(9, 14).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(10, 13).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(10, 1).getProvidedInterfaces().get(0)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getAuthenticationComponent(11)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getAuthenticationComponent(12)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getAuthenticationComponent(13)));
	
	}
 else if (runCount == 4)
		// 30 CF
		{
			System.out.print("\n 4th mape injects 30 more CF");
			this.updateReliabiliy_G_1();
			Random rnd;
            int compnt=0;
			for (int shop = 14; shop < 24; shop++)// 10

			{    if (compnt>17)
				 compnt=0;
				//rnd = new Random();
				//int compnt = rnd.nextInt(17);
				injections.add(new Injection<Component>(IssueType.CF1, this.getComponent(shop, compnt)));
				compnt++;
				
			}
			for (int shop = 24; shop < 34; shop++)// 10
			{if (compnt>17)
				 compnt=0;
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.getComponent(shop, compnt).getProvidedInterfaces().get(0)));
				compnt++;
			}
			for (int shop = 34; shop < 44; shop++)// 10
			{ if (compnt>17)
				 compnt=0;
				//rnd = new Random();
				//int compnt = rnd.nextInt(17);
				injections.add(new Injection<Component>(IssueType.CF3, this.getComponent(shop, compnt)));
				compnt++;
			}
		} else if (runCount == 5)
	// 50 CF
		{System.out.print("\n 5th mape injects 50 more CF");
		this.updateReliabiliy_G_1();
		//Random rnd;
		int compnt=0;
		for (int shop = 44; shop < 61; shop++)// 17

		{if (compnt>17)
			 compnt=0;
			//rnd = new Random();
			//int compnt = rnd.nextInt(17);
			injections.add(new Injection<Component>(IssueType.CF1, this.getComponent(shop, compnt)));
			compnt++;
		}
		for (int shop = 61; shop < 78; shop++)// 10
		{
			if (compnt>17)
				 compnt=0;
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.getComponent(shop, compnt).getProvidedInterfaces().get(0)));
				compnt++;
		}
		for (int shop = 78; shop < 94; shop++)// 10
		{
			//rnd = new Random();
			//int compnt = rnd.nextInt(17);
			if (compnt>17)
				 compnt=0;
			injections.add(new Injection<Component>(IssueType.CF3, this.getComponent(shop, compnt)));
			 compnt++;
		}
	
		} else if (runCount == 6)
	// 100 CF
		{System.out.print("\n 6th mape injects 100 more CF");
		Random rnd;
		int compnt=0;
		for (int shop = 0; shop < 33; shop++)// 33

		{	
			//rnd = new Random();
			//int compnt = rnd.nextInt(17);
		if (compnt>17)
			 compnt=0;
			injections.add(new Injection<Component>(IssueType.CF3, this.getComponent(shop, compnt)));
			compnt++;
		}
		for (int shop = 33; shop < 66; shop++)// 33
		{
			if (compnt>17)
				 compnt=0;
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.getComponent(shop, compnt).getProvidedInterfaces().get(0)));
				compnt++;
		}
		for (int shop = 66; shop < 100; shop++)// 10
		{
			//rnd = new Random();
			//int compnt = rnd.nextInt(17);
			if (compnt>17)
				 compnt=0;
			injections.add(new Injection<Component>(IssueType.CF1, this.getComponent(shop, compnt)));if (compnt>17)
				 compnt++;
		}
		
		} else if (runCount == 7)
	// 7 CF
		{System.out.print("\n 7th mape injects 7 more CF");
		this.updateReliabiliy_G_1();
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.findMostCriticalComponent(6).getProvidedInterfaces()
				.get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(6,1).getProvidedInterfaces()
				.get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(6,2).getProvidedInterfaces()
				.get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(6,5).getProvidedInterfaces()
				.get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(6,8).getProvidedInterfaces()
				.get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(6,9).getProvidedInterfaces()
				.get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(6,10).getProvidedInterfaces()
				.get(0)));
		
		
		}else if (runCount == 8)
	// 12 CF
		{System.out.print("8th mape injects 12 more CF");
		for (int i=1;i<=12;i++){
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(7, i)));}
		
		
		}else if (runCount == 9)
	// 7 CF
		{System.out.print("9th mape injects 7 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(8, 13)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(8, 5)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(8, 6)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(8, 12)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(8,10).getProvidedInterfaces()
				.get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(8,1).getProvidedInterfaces()
				.get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(8,2).getProvidedInterfaces()
				.get(0)));
		
		
		
		}else if (runCount == 10)
	//  
		{System.out.print("10th mape injects 1  more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(8,8)));
	
		
		
		}else if (runCount == 11)
	// 4 CF
		{System.out.print("11th mape injects 4 more CF");
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(9, 1)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(1, 3)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.findMostCriticalComponent(9)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(2, 5)));
		
		
		
		}else if (runCount == 12)
	//  19 CF
		{
			System.out.print("12th mape injects 19 more CF");
			injections.add(new Injection<Component>(IssueType.CF1, this.getComponent(10, 1)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					this.getComponent(10, 2).getProvidedInterfaces().get(0)));
			injections.add(new Injection<Component>(IssueType.CF1, this.getComponent(10, 3)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					this.getComponent(10, 4).getProvidedInterfaces().get(0)));
			injections.add(new Injection<Component>(IssueType.CF1, this.getComponent(10, 5)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					this.getComponent(10, 6).getProvidedInterfaces().get(0)));
			injections.add(new Injection<Component>(IssueType.CF1, this.getComponent(10, 7)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					this.getComponent(10, 8).getProvidedInterfaces().get(0)));
			injections.add(new Injection<Component>(IssueType.CF1, this.getComponent(10, 9)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					this.getComponent(10, 10).getProvidedInterfaces().get(0)));
			injections.add(new Injection<Component>(IssueType.CF1, this.getComponent(10, 11)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					this.getComponent(10, 12).getProvidedInterfaces().get(0)));
			injections.add(new Injection<Component>(IssueType.CF1, this.getComponent(10, 13)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					this.getComponent(10, 14).getProvidedInterfaces().get(0)));
			injections.add(new Injection<Component>(IssueType.CF1, this.getComponent(10, 15)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					this.getComponent(10, 16).getProvidedInterfaces().get(0)));
			injections.add(new Injection<Component>(IssueType.CF1, this.getComponent(10, 17)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					this.getComponent(9, 13).getProvidedInterfaces().get(0)));
			injections.add(new Injection<Component>(IssueType.CF1, this.getComponent(9, 12)));
																				
		
		
		}else if (runCount == 13)
	// 5  CF
		{System.out.print("13th mape injects 5 more CF");
		
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(19, 1)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(21, 13).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(20, 14).getProvidedInterfaces().get(0)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(11,16)));
		injections.add(new Injection<Component>(IssueType.CF3, this
			.getComponent(19,17)));
		
		
		
		}else if (runCount == 14)
	// 3 CF
		{System.out.print("14th mape injects 3 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(11, 1)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(11,2)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(11, 3).getProvidedInterfaces().get(0)));
		
		
		
		}else if (runCount == 15)
	// 1 CF
		{System.out.print("15th mape injects 1 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(11, 4)));
		
		
		
		}else if (runCount == 16)
	// 10 CF
		{System.out.print("16th mape injects 10 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(11, 5)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(11, 7)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(11, 6)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(11, 8)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(11, 9)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(11, 10).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(11, 11).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(11, 12).getProvidedInterfaces().get(0)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(11,13)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(11,14)));
		
		}else if (runCount == 17)
	// 11 CF 
		{System.out.print("17 th mape injects 11 more CF");
		
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(12, 5)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(12, 7)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(12, 6)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(12, 8)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(12, 9)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(12, 10).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(12, 11).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(12, 12).getProvidedInterfaces().get(0)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(12,13)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(12,14)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(12,4)));


		
		
		}else if (runCount == 18)
	// 6 CF
		{System.out.print("18th mape injects 6 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(43, 1)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(15, 2)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(15, 3)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(81, 15).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(15, 16).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(14, 17).getProvidedInterfaces().get(0)));
		
		}else if (runCount == 19)
	// 22 CF
		{System.out.print("19th mape injects 22 more CF");
		this.updateReliabiliy_TW_1();
		this.updateReliabiliy_G_1();
		this.updateReliabiliy_FB_1();
              for (int i=0; i<18; i=i+3){		
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(13, i)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(13, i+1).getProvidedInterfaces().get(0)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(13,i+2)));}
        injections.add(new Injection<Component>(IssueType.CF1, this
    				.getComponent(14, 1)));
        injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
    				this.getComponent(14, 2).getProvidedInterfaces().get(0)));
        injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(14, 3).getProvidedInterfaces().get(0)));
        injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(14,4)));
          	
              

		
		
		}else if (runCount == 20)
	// 10 CF
		{System.out.print("20th mape injects 10 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(14, 6)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(14, 5)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(14, 7)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(14, 8)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(14, 9).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(14, 10).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(14, 11).getProvidedInterfaces().get(0)));
	       injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(14,12)));
	       injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(14,13)));
	       injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(14,14)));
		
		
		}else if (runCount == 21)
	// 105 CF
		{System.out.print("21st mape injects 105 more CF");
	
       for(int j=15; j<20; j++){ // 5 tenants
    	   
		for (int i=0; i<18; i=i+3){	// 18 components 
	
	injections.add(new Injection<Component>(IssueType.CF1, this
			.getComponent(j, i)));
	injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
			this.getComponent(j, i+1).getProvidedInterfaces().get(0)));
	injections.add(new Injection<Component>(IssueType.CF3, this
			.getComponent(j,i+2)));}
       }//5*18=90 Cf so far
       for (int i=0; i<15; i=i+3){	// 15 more CF
    		
    		injections.add(new Injection<Component>(IssueType.CF1, this
    				.getComponent(20, i)));
    		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
    				this.getComponent(20, i+1).getProvidedInterfaces().get(0)));
    		injections.add(new Injection<Component>(IssueType.CF3, this
    				.getComponent(20,i+2)));}
       
	
		
		}else if (runCount == 22)
	// 7 CF
		{System.out.print("22nd mape injects 7 more CF");
		
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(21, 1)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(21, 2)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(21, 3)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(21, 4)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(21, 5).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(21, 6).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(21, 7).getProvidedInterfaces().get(0)));
		
		
		}else if (runCount == 23)
	// 26 CF
		{System.out.print("23rd mape injects 26 more CF");
		
        for (int i=0; i<18; i=i+3){		
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(22, i)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(22, i+1).getProvidedInterfaces().get(0)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(22,i+2)));} //18 CF
       injections.add(new Injection<Component>(IssueType.CF1, this
   				.getComponent(21, 8)));
       injections.add(new Injection<Component>(IssueType.CF1, this
  				.getComponent(21, 9)));
       injections.add(new Injection<Component>(IssueType.CF1, this
  				.getComponent(21, 10)));
       injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
   				this.getComponent(21, 12).getProvidedInterfaces().get(0)));
       injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
  				this.getComponent(21, 11).getProvidedInterfaces().get(0)));
       injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(21, 13).getProvidedInterfaces().get(0)));
       injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(21,14)));
       injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(21,15)));
		
		
		}else if (runCount == 24)
	// 442 CF
		{System.out.print("24th mape injects 442 more CF");

	       for(int j=23; j<47; j++){ // 24 tanents
	    	   
			for (int i=0; i<18; i=i+3){	// 18 components 
		
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(j, i)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
				this.getComponent(j, i+1).getProvidedInterfaces().get(0)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(j,i+2)));}
	       }//24*18=432 Cf so far
	       
	       for (int i=0; i<3; i=i+3){	// 3 components 
	   		
	   		injections.add(new Injection<Component>(IssueType.CF1, this
	   				.getComponent(47, i)));
	   		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
	   				this.getComponent(47, i+1).getProvidedInterfaces().get(0)));
	   		injections.add(new Injection<Component>(IssueType.CF3, this
	   				.getComponent(47,i+2)));} // 432+9=441 cf so far
	       injections.add(new Injection<Component>(IssueType.CF1, this
	   				.getComponent(47, 15)));
	   	       
		
		
		}else if (runCount == 25)
	//   1CF
		{System.out.print("25th mape injects 1 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(47, 16)));
		
		
		
		}else if (runCount == 26)
	// 5 CF
		{System.out.print("26th mape injects 5 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(48, 1)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(48, 2)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(48, 3)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(48, 4)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(48, 5).getProvidedInterfaces().get(0)));
		
		
		}else if (runCount == 27)
	// 29 CF
		{System.out.print("27th mape injects 29 more CF");
		this.updateReliabiliy_G_2();
		this.updateReliabiliy_FB_2();
		this.updateReliabiliy_TW_2();
		for (int i=0;i<14;i=i+2){
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(49, i)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(50, i)));//14 CF1
			
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.getComponent(49, i+1).getProvidedInterfaces().get(0)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.getComponent(50, i+1).getProvidedInterfaces().get(0)));// 14 CF2
			}
			
		
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getAuthenticationComponent(51)));
		
		
		}else if (runCount == 28)
	// 2 CF
		{System.out.print("28th mape injects 2 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(48, 13)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(48, 14)));
		
		
		}else if (runCount == 29)
	// 41 CF
		{System.out.print("29th mape injects 41 more CF");
		this.updateReliabiliy_G_3();
		this.updateReliabiliy_TW_3();
		this.updateReliabiliy_FB_3();
		  for(int j=52; j<54; j++){ // 2 tanents
	    	   
				for (int i=0; i<18; i=i+3){	// 18 components 
			
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(j, i)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					this.getComponent(j, i+1).getProvidedInterfaces().get(0)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(j,i+2)));}
		       }//2*18=36 Cf so far
		  injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(54, 1)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(54, 8)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(54, 3)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(54, 14)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,this
					.getComponent(54, 5).getProvidedInterfaces().get(0)));
		
		
		}else if (runCount == 30)
	// 15 CF
		{System.out.print("30th mape injects 15  more CF");
            for (int i=0; i<15; i=i+3){	// 15 components 
			
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(55, i)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					this.getComponent(55, i+1).getProvidedInterfaces().get(0)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(55,i+2)));}
		
		
		}else if (runCount == 31)
	// 2 CF
		{System.out.print("31st mape injects 2 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(56, 13)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(56, 5)));
		
		
		
		}else if (runCount == 32)
	//4  CF
		{System.out.print("32nd mape injects 4  more CF");
		this.updateReliabiliy_G_3();
		this.updateReliabiliy_TW_3();
		this.updateReliabiliy_FB_3();
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(56, 1)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(56, 2)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(56, 3)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,this
				.getComponent(56, 4).getProvidedInterfaces().get(0)));
		
		
		}else if (runCount == 33)
	// 15 CF
		{System.out.print("33rd mape injects 15 more CF");
		   for (int i=0; i<15; i=i+3){	// 15 components 
				
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(57, i)));
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
						this.getComponent(57, i+1).getProvidedInterfaces().get(0)));
				injections.add(new Injection<Component>(IssueType.CF3, this
						.getComponent(57,i+2)));}
		
		
		}else if (runCount == 34)
	// 5 CF
		{System.out.print("34th mape injects 5 more CF");
		this.updateReliabiliy_G_4();
		this.updateReliabiliy_FB_4();
		this.updateReliabiliy_TW_4();
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(58, 1)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(58, 2)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(58, 3)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(58, 4)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,this
				.getComponent(58,5).getProvidedInterfaces().get(0)));
		
		}else if (runCount == 35)
	// 3 CF
		{System.out.print("35th mape injects 3 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(58, 6)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(58, 7)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2,this
		        .getComponent(58, 8).getProvidedInterfaces().get(0)));
		
		
		}else if (runCount == 36)
	// 62 CF
		{System.out.print("36th mape injects 62 more CF");
		
		  for(int j=59; j<62; j++){ // 3 tanents
	    	   
				for (int i=0; i<18; i=i+3){	// 18 components 
			
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(j, i)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					this.getComponent(j, i+1).getProvidedInterfaces().get(0)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(j,i+2)));}
		       }//3*18=54 Cf so far
		  injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(58, 9)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(58, 10)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(58, 11)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(58, 12)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,this
					.getComponent(58, 13).getProvidedInterfaces().get(0)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,this
					.getComponent(58, 14).getProvidedInterfaces().get(0)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,this
					.getComponent(58, 15).getProvidedInterfaces().get(0)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,this
					.getComponent(58, 16).getProvidedInterfaces().get(0)));
		
		
		
		}else if (runCount == 37)
	// 2 CF
		{System.out.print("37th mape injects 2 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(62, 1)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(62, 2)));
		
		
		}else if (runCount == 38)
	// 1 CF
		{System.out.print("38th mape injects 1 more CF");
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(62, 3)));
	
		
		
		}else if (runCount == 39)
	// 19 CF
		{System.out.print("39th mape injects 19 more CF");
		this.updateReliabiliy_TW_1();
		this.updateReliabiliy_G_1();
		this.updateReliabiliy_FB_1();
		  for (int i=0; i<18; i=i+3){	// 18 components 
				
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(63, i)));
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
						this.getComponent(63, i+1).getProvidedInterfaces().get(0)));
				injections.add(new Injection<Component>(IssueType.CF3, this
						.getComponent(63,i+2)));}
		  injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(62, 4)));
		
		
		}else if (runCount == 40)
	// 29 CF
		{System.out.print("40th mape injects 29  more CF");
		for (int i=0;i<14;i=i+2){
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(64, i)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(65, i)));//14 CF1
			
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.getComponent(64, i+1).getProvidedInterfaces().get(0)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.getComponent(65, i+1).getProvidedInterfaces().get(0)));// 14 CF2
			}
			
		
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(65,16)));
		
		
		
		}else if (runCount == 41)
	// 21 CF
		{System.out.print("41st mape injects 21 more CF");
		for (int i=0;i<10;i=i+2){
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(66, i)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(67, i)));//10 CF1
			
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.getComponent(66, i+1).getProvidedInterfaces().get(0)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.getComponent(67, i+1).getProvidedInterfaces().get(0)));// 10 CF2
			}
			
		
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getAuthenticationComponent(68)));
		
		
		}else if (runCount == 42)
	// 9 CF
		{System.out.print("42nd mape injects 9 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(69, 1)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(69, 5)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(69, 6)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(69, 2)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(69, 3)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(69, 4)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(69, 7).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(69, 8).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(69, 9).getProvidedInterfaces().get(0)));
		
		
		}else if (runCount == 43)
	// 8 CF
		{System.out.print("43rd mape injects 8 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(70, 10)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(70, 11)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(70, 12)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(70, 13)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(70, 14)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(70, 15).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(70, 16).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(70, 17).getProvidedInterfaces().get(0)));
		
		
		}else if (runCount == 44)
	// 1 CF
		{System.out.print("44th mape injects 1 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this.getComponent(71, 1)));
				
	
		
		
		}else if (runCount == 45)
	// 7 CF
		{System.out.print("45th mape injects 7 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(71, 10)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(71, 11)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(71, 12)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(71, 13)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(71, 14)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(71, 15).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(71, 16).getProvidedInterfaces().get(0)));
		
		
		
		}else if (runCount == 46)
	// 12 CF
		{System.out.print("46th mape injects 12 more CF");
		this.updateReliabiliy_TW_1();
		this.updateReliabiliy_G_1();
		this.updateReliabiliy_FB_1();
		  for (int i=0; i<4; i=i+3){	// 4 components 
				
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(72, i)));
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
						this.getComponent(72, i+1).getProvidedInterfaces().get(0)));
				injections.add(new Injection<Component>(IssueType.CF3, this
						.getComponent(72,i+2)));}
		
		
		}else if (runCount == 47)
	// 8 CF
		{System.out.print("47th mape injects 8 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(91, 8)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(82, 1)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(83, 2)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(83, 3)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(84, 4)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(71, 5).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(71, 6).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(71, 7).getProvidedInterfaces().get(0)));
		
		
		}else if (runCount == 48)
	// 5 CF
		{System.out.print("48th mape injects 5 more CF");
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(73, 13)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(73, 5)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(73, 6)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(73, 12)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(73, 7).getProvidedInterfaces().get(0)));
		
		
		}else if (runCount == 49)
	// 41 CF
		{System.out.print("49th mape injects 41 more CF");
		  for(int j=74; j<76; j++){ // 2 tanents
	    	   
				for (int i=0; i<18; i=i+3){	// 18 components 
			
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(j, i)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					this.getComponent(j, i+1).getProvidedInterfaces().get(0)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(j,i+2)));}
		       }//2*18=36 Cf so far
		  injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(76, 9)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(76, 10)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,this
					.getComponent(76, 13).getProvidedInterfaces().get(0)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,this
					.getComponent(76, 14).getProvidedInterfaces().get(0)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,this
					.getComponent(76, 15).getProvidedInterfaces().get(0)));

		

		
		}else if (runCount == 50)
	// 23 CF
		{System.out.print("50th mape injects 23 more CF");
		this.updateReliabiliy_TW_1();
		this.updateReliabiliy_G_1();
		this.updateReliabiliy_FB_1();
		for (int i=0;i<10;i=i+2){
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(77, i)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(78, i)));//10 CF1
			
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.getComponent(77, i+1).getProvidedInterfaces().get(0)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.getComponent(78, i+1).getProvidedInterfaces().get(0)));// 10 CF2
			}
			
		
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getAuthenticationComponent(79)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getAuthenticationComponent(80)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.findMostCriticalComponent(81)));
		
		}	*/
	return injections;
}

@Override
public void notifyAboutSuccess(
		List<Injection<? extends ArchitecturalElement>> injections) {
	for (Injection<? extends ArchitecturalElement> i : injections) {
		if (!i.isSuccess()) {
			String message = "The simulator could not successfully inject a "
					+ i.getIssueType()
					+ " issue to element "
					+ i.getTarget();
			System.err.println(message);
			throw new RuntimeException(message);
		}
	}
}
private Component getComponent(int tenantNumber, int componentNumber) {
	Tenant tenant = this.eArchitecture.getTenants().get(tenantNumber);
	return tenant.getComponents().get(componentNumber);
}

private Component getAuthenticationComponent(int tenantNumber) {
	Tenant tenant = this.eArchitecture.getTenants().get(tenantNumber);
	Component authenticationService = null;
	for (Component c : tenant.getComponents()) {
		if (c.getType().getName().equals("Authentication Service")) {
			authenticationService = c;
			break;
		}
	}
	return authenticationService;
}

private void updateReliabiliy_TW_1() {
	for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
		String componentTypeName = ct.getName();
		if (componentTypeName.startsWith("Twitter")) {
			ct.setReliability(0.9);
		}
	}
}

private void updateReliabiliy_G_1() {
	for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
		String componentTypeName = ct.getName();
		if (componentTypeName.startsWith("Google")) {
			ct.setReliability(0.92);
		}
	}
}

private void updateReliabiliy_FB_1() {
	for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
		String componentTypeName = ct.getName();
		if (componentTypeName.startsWith("Facebook")) {
			ct.setReliability(0.65);
		}
	}
}
private void updateReliabiliy_TW_2() {
	for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
		String componentTypeName = ct.getName();
		if (componentTypeName.startsWith("Twitter")) {
			ct.setReliability(0.54);
		}
	}
}

private void updateReliabiliy_G_2() {
	for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
		String componentTypeName = ct.getName();
		if (componentTypeName.startsWith("Google")) {
			ct.setReliability(0.91);
		}
	}
}

private void updateReliabiliy_FB_2() {
	for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
		String componentTypeName = ct.getName();
		if (componentTypeName.startsWith("Facebook")) {
			ct.setReliability(0.73);
		}
	}
}
private void updateReliabiliy_TW_3() {
	for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
		String componentTypeName = ct.getName();
		if (componentTypeName.startsWith("Twitter")) {
			ct.setReliability(0.64);
		}
	}
}

private void updateReliabiliy_G_3() {
	for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
		String componentTypeName = ct.getName();
		if (componentTypeName.startsWith("Google")) {
			ct.setReliability(0.79);
		}
	}
}

private void updateReliabiliy_FB_3() {
	for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
		String componentTypeName = ct.getName();
		if (componentTypeName.startsWith("Facebook")) {
			ct.setReliability(0.5);
		}
	}
}
private void updateReliabiliy_TW_4() {
	for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
		String componentTypeName = ct.getName();
		if (componentTypeName.startsWith("Twitter")) {
			ct.setReliability(0.47);
		}
	}
}

private void updateReliabiliy_G_4() {
	for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
		String componentTypeName = ct.getName();
		if (componentTypeName.startsWith("Google")) {
			ct.setReliability(0.88);
		}
	}
}

private void updateReliabiliy_FB_4() {
	for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
		String componentTypeName = ct.getName();
		if (componentTypeName.startsWith("Facebook")) {
			ct.setReliability(1.3);
		}
	}
}

private Component findMostCriticalComponent(int tenantNumber) {
	Tenant tenant = this.eArchitecture.getTenants().get(tenantNumber);
	double selectedCriticality = Double.MIN_VALUE;
	Component selectedComponent = null;
	for (Component component : tenant.getComponents()) {
		double criticality = component.getType().getCriticality();
		if (criticality > selectedCriticality) {
			selectedCriticality = criticality;
			selectedComponent = component;
		}

	}
	return selectedComponent;
}

private Component findLeastCriticalComponent(int tenantNumber) {
	Tenant tenant = this.eArchitecture.getTenants().get(tenantNumber);
	double selectedCriticality = Double.MAX_VALUE;
	Component selectedComponent = null;
	for (Component component : tenant.getComponents()) {
		double criticality = component.getType().getCriticality();
		if (criticality < selectedCriticality) {
			selectedCriticality = criticality;
			selectedComponent = component;
		}
	}
	return selectedComponent;
}





}