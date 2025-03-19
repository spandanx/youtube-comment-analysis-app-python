/*
SQLyog Community v13.3.0 (64 bit)
MySQL - 8.4.3 : Database - stock_db
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
CREATE DATABASE /*!32312 IF NOT EXISTS*/`stock_db` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;

USE `stock_db`;

/*Table structure for table `yt_comm_user` */

DROP TABLE IF EXISTS `yt_comm_user`;

CREATE TABLE `yt_comm_user` (
  `username` varchar(40) NOT NULL,
  `full_name` text,
  `email` varchar(100) DEFAULT NULL,
  `hashed_password` text,
  `disabled` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

/*Data for the table `yt_comm_user` */

insert  into `yt_comm_user`(`username`,`full_name`,`email`,`hashed_password`,`disabled`) values 
('admin','Admin User','admin@admin.com','$2b$12$G6Qw5e.K871doase2mJqgepPaB7frMIWb973E9zspNl3dNrHSik8C',0);

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
