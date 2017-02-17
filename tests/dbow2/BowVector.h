/**
 * File: BowVector.h
 * Date: March 2011
 * Author: Dorian Galvez-Lopez
 * Description: bag of words vector
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_BOW_VECTOR__
#define __D_T_BOW_VECTOR__

#include <iostream>
#include <map>
#include <vector>

namespace DBoW2 {

/// Id of words
typedef unsigned int WordId;

/// Value of a word
typedef double WordValue;

/// Id of nodes in the vocabulary treee
typedef unsigned int NodeId;

/// L-norms for normalization
enum LNorm
{
    L1,
    L2
};

/// Weighting type
enum WeightingType
{
    TF_IDF,
    TF,
    IDF,
    BINARY
};

/// Scoring type
enum ScoringType
{
    L1_NORM,
    L2_NORM,
    CHI_SQUARE,
    KL,
    BHATTACHARYYA,
    DOT_PRODUCT,
};

/// Vector of words to represent images
class BowVector: 
        public std::map<WordId, WordValue>
{
public:

    /**
     * Constructor
     */
    BowVector(void){}

    /**
     * Destructor
     */
    ~BowVector(void){}

    /**
     * Adds a value to a word value existing in the vector, or creates a new
     * word with the given value
     * @param id word id to look for
     * @param v value to create the word with, or to add to existing word
     */
    void addWeight(WordId id, WordValue v){
        BowVector::iterator vit = this->lower_bound(id);

        if(vit != this->end() && !(this->key_comp()(id, vit->first)))
        {
            vit->second += v;
        }
        else
        {
            this->insert(vit, BowVector::value_type(id, v));
        }
    }

    /**
     * Adds a word with a value to the vector only if this does not exist yet
     * @param id word id to look for
     * @param v value to give to the word if this does not exist
     */
    void addIfNotExist(WordId id, WordValue v)
    {
        BowVector::iterator vit = this->lower_bound(id);

        if(vit == this->end() || (this->key_comp()(id, vit->first)))
        {
            this->insert(vit, BowVector::value_type(id, v));
        }
    }

    /**
     * L1-Normalizes the values in the vector
     * @param norm_type norm used
     */
    void normalize(LNorm norm_type)
    {
        double norm = 0.0;
        BowVector::iterator it;

        if(norm_type == DBoW2::L1)
        {
            for(it = begin(); it != end(); ++it)
                norm += fabs(it->second);
        }
        else
        {
            for(it = begin(); it != end(); ++it)
                norm += it->second * it->second;
            norm = sqrt(norm);
        }

        if(norm > 0.0)
        {
            for(it = begin(); it != end(); ++it)
                it->second /= norm;
        }
    }


    /**
     * Prints the content of the bow vector
     * @param out stream
     * @param v
     */
    friend std::ostream& operator<<(std::ostream &out, const BowVector &v)
    {
        BowVector::const_iterator vit;
        std::vector<unsigned int>::const_iterator iit;
        unsigned int i = 0;
        const unsigned int N = v.size();
        for(vit = v.begin(); vit != v.end(); ++vit, ++i)
        {
            out << "<" << vit->first << ", " << vit->second << ">";

            if(i < N-1) out << ", ";
        }
        return out;
    }

    /**
     * Saves the bow vector as a vector in a matlab file
     * @param filename
     * @param W number of words in the vocabulary
     */
    void saveM(const std::string &filename, size_t W) const
    {
        std::fstream f(filename.c_str(), std::ios::out);

        WordId last = 0;
        BowVector::const_iterator bit;
        for(bit = this->begin(); bit != this->end(); ++bit)
        {
            for(; last < bit->first; ++last)
            {
                f << "0 ";
            }
            f << bit->second << " ";

            last = bit->first + 1;
        }
        for(; last < (WordId)W; ++last)
            f << "0 ";

        f.close();
    }
};

} // namespace DBoW2

#endif
