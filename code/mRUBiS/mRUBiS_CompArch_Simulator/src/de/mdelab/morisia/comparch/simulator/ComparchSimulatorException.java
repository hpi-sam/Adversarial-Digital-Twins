package de.mdelab.morisia.comparch.simulator;

/**
 * Exception if something goes wrong during a simulation.
 * 
 * @author thomas vogel
 * @version 0.30
 *
 */
public class ComparchSimulatorException extends RuntimeException {

	/**
	 * Serial version uid.
	 */
	private static final long serialVersionUID = -5553200657798718406L;

	/**
	 * Constructor.
	 * 
	 * @param msg
	 *            failure message.
	 */
	public ComparchSimulatorException(String msg) {
		super(msg);
	}

	/**
	 * Constructor.
	 * 
	 * @param msg
	 *            failure message.
	 * @param cause
	 *            the cause for raising this exception.
	 */
	public ComparchSimulatorException(String msg, Throwable cause) {
		super(msg, cause);
	}

}
