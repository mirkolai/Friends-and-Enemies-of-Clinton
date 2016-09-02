SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `Semeval_Task_6_stance_detection`
--

-- --------------------------------------------------------

--
-- Struttura della tabella `data_test`
--

CREATE TABLE IF NOT EXISTS `data_test` (
  `ID` int(11) NOT NULL AUTO_INCREMENT,
  `Tweet` varchar(142) DEFAULT NULL,
  `Target` varchar(32) DEFAULT NULL,
  `Stance` varchar(7) DEFAULT NULL,
  `Opinion Towards` varchar(125) DEFAULT NULL,
  `Sentiment` varchar(5) DEFAULT NULL,
  PRIMARY KEY (`ID`)
) ENGINE=MyISAM  DEFAULT CHARSET=utf8 AUTO_INCREMENT=1957 ;

-- --------------------------------------------------------

--
-- Struttura della tabella `data_training`
--

CREATE TABLE IF NOT EXISTS `data_training` (
  `ID` int(11) NOT NULL AUTO_INCREMENT,
  `Tweet` varchar(142) DEFAULT NULL,
  `Target` varchar(32) DEFAULT NULL,
  `Stance` varchar(7) DEFAULT NULL,
  `Opinion Towards` varchar(125) DEFAULT NULL,
  `Sentiment` varchar(5) DEFAULT NULL,
  PRIMARY KEY (`ID`)
) ENGINE=MyISAM  DEFAULT CHARSET=utf8 AUTO_INCREMENT=2930 ;

-- --------------------------------------------------------

--
-- Struttura della tabella `data_trial`
--

CREATE TABLE IF NOT EXISTS `data_trial` (
  `ID` int(3) NOT NULL DEFAULT '0',
  `Target` varchar(24) DEFAULT NULL,
  `Tweet` varchar(137) DEFAULT NULL,
  `Stance` varchar(7) DEFAULT NULL,
  PRIMARY KEY (`ID`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
