//******************************************************************
// 
//  Generated by RoboCompDSL
//  
//  File name: ByteSequencePublisher.ice
//  Source: ByteSequencePublisher.idsl
//
//******************************************************************
#ifndef ROBOCOMPBYTESEQUENCEPUBLISHER_ICE
#define ROBOCOMPBYTESEQUENCEPUBLISHER_ICE
module RoboCompByteSequencePublisher
{
	sequence <byte> bytesequence;
	interface ByteSequencePublisher
	{
		void newsequence (bytesequence bs);
	};
};

#endif
