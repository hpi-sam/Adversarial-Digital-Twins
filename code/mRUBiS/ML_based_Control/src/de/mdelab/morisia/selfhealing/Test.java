package de.mdelab.morisia.selfhealing;

import java.util.Arrays;
import java.util.Random;
import java.util.logging.Level;

import org.eclipse.emf.ecore.resource.Resource;

import de.mdelab.morisia.comparch.Annotations;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.ComparchFactory;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.simulator.Capability;
import de.mdelab.morisia.comparch.simulator.ComparchSimulator;
import de.mdelab.morisia.comparch.simulator.Injection;
import de.mdelab.morisia.comparch.simulator.InjectionStrategy;
import de.mdelab.morisia.comparch.simulator.IssueType;
import de.mdelab.morisia.selfhealing.incremental.EventListener;
import mRUBiS.Observations.Observations;

public class Test {
   public static void main(String[] args) {
	
	   
		  int R=29;
		int run= Math.floorMod(R,101);
		 System.out.println("\n ============================================================================= CF size  " +run);
		int p=Math.floorDiv(R,100);
		 System.out.println("\n shop starts at  " +p);
		int shop=run+p;
		int cfType=3;
		int extra=Math.floorMod(run, cfType);
		 System.out.println(" extra =" + extra);
		int interval=Math.floorDiv(run, cfType);
		 System.out.println("interval =" + interval);
	    int  f1=interval;
	     int f2=interval;
		//int f3=interval;
		int t=109; // t=x;
		int comp=shop+t;
		int c1=0;
		int c2 =0;
		int c3=0;
		//int c5 =0;
	int f3=interval+extra;
		
		
		
		while(c1<f1)
		{	c1++;
			if(shop>99)
			shop = Math.floorMod(shop, 100);
			 comp++;
				if(comp>17)
					comp = Math.floorMod(comp, 18);
				  System.out.println("\n CF1  shop:  "+ shop + "  Comp:  "+comp+"\n");
			shop++;
		}	
		 System.out.println(" ___________________________________         f1 "+ c1 );

		while(c2<f2)
		{	c2++;
			if(shop>99)
			shop = Math.floorMod(shop, 100);
			 comp++;
				if(comp>17)
					comp = Math.floorMod(comp, 18);
				  System.out.println("\n CF2  shop:  "+ shop + "  Comp:  "+comp+"\n");
			shop++;
		}
		 System.out.println(" ___________________________________         f2 "+ c2);

		while(c3<f3)
		{	c3++;
			if(shop>99)
			shop = Math.floorMod(shop, 100);
			 comp++;
				if(comp>17)
					comp = Math.floorMod(comp, 18);
				  System.out.println("\n CF3  shop:  "+ shop + "  Comp:  "+comp+"\n");
			shop++;
		}
		 System.out.println(" ___________________________________         f3 "+ c3);
	/*	while(c5<f5)
		{	c5++;
			if(shop>99)
			shop = Math.floorMod(shop, 100);
			 comp++;
				if(comp>17)
					comp = Math.floorMod(comp, 18);
				  System.out.println("\n CF5  shop:  "+ shop + "  Comp:  "+comp+"\n");
			shop++;
		}
		 System.out.println(" ___________________________________         f5 "+ c5);	
   
		 
		 
		 */
	/*
		//CF1
		for (shop=0; shop<f1;shop++)
		{ comp++;
		if(comp>17)
			comp = Math.floorMod(comp, 18);
		  System.out.println("\n CF1  shop:  "+ shop + "  Comp:  "+comp+"\n");
		}
		 System.out.println(" ___________________________________         f1 "+ f1 + "   comp  "+ comp);
		//CF2
		for (shop=f1; shop<f2;shop++)
		{comp++;
		// System.out.println( " \n  comp  "+ comp);
		if(comp>17)
			comp = Math.floorMod(comp, 18);
	//	 System.out.println( " \n  comp  "+ comp);
		 System.out.println("\n CF2  shop:  "+ shop + "  Comp:  "+comp+"\n");
		}
		 System.out.println(" ___________________________________         f2 "+ f2);
		//CF3
		for (shop=f2; shop<f3;shop++)
		{comp++;
		if(comp>17)
			comp = Math.floorMod(comp, 18);
		 System.out.println("\n CF3  shop:  "+ shop + "  Comp:  "+comp+"\n");
		}
		 System.out.println(" ___________________________________        f3 "+ f3);
	  
			for (shop=f3; shop<run;shop++)
			{comp++;
			if(shop>17)
				comp = Math.floorMod(comp, 18);
			 System.out.println("\n CF5  shop:  "+ shop + "  Comp:  "+comp+"\n");
			}	
			// System.out.println(" ___________________________________         cf size "+ shop);
			*/
   }
}